from __future__ import annotations

import logging
import math
import random
import socket
import time
from dataclasses import replace
from contextlib import nullcontext
from multiprocessing import Value
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from restormer_teacher.checkpointing import (
    checkpoint_teacher_metadata,
    is_legacy_teacher_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from restormer_teacher.config import TeacherResolvedConfig, resolve_teacher_config
from restormer_teacher.data import (
    collect_train_pairs,
    collect_val_pairs,
    make_dataloaders,
    summarize_pairs,
    validate_data_layout,
)
from restormer_teacher.ema import ModelEMA
from restormer_teacher.logging_utils import (
    append_jsonl,
    atomic_write_json,
    load_history,
    load_jsonl,
    merge_history_meta,
    setup_logging,
    wall_ts,
    write_json,
)
from restormer_teacher.losses import TeacherCompositeLoss, residual_supervision_l1
from restormer_teacher.metrics import residual_l1_ratio, tensor_psnr
from restormer_teacher.model import build_teacher_model
from restormer_teacher.run_state import reconcile_run_state

logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_batch_size_auto(device: torch.device) -> int:
    if device.type != "cuda":
        return 2
    total = torch.cuda.get_device_properties(0).total_memory
    # ~16GiB threshold
    if total <= 17 * (1024**3):
        return 4
    return 8


def _resolve_batch_size(setting: int | str | None, device: torch.device, *, smoke_test: bool) -> int:
    if setting == "auto" or setting is None:
        batch_size = _pick_batch_size_auto(device)
    else:
        batch_size = int(setting)
    if smoke_test:
        batch_size = min(batch_size, 2)
    return batch_size


def _autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.detach().float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _save_val_samples(
    run_root: Path,
    epoch: int,
    lr: torch.Tensor,
    hr: torch.Tensor,
    pred_ema: torch.Tensor,
    max_k: int = 4,
) -> Path:
    out_dir = run_root / "val_samples" / f"epoch_{epoch:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    b = min(lr.shape[0], max_k)
    scale = 8.0
    for i in range(b):
        _tensor_to_pil(lr[i]).save(out_dir / f"{i:02d}_lr.png")
        _tensor_to_pil(hr[i]).save(out_dir / f"{i:02d}_hr.png")
        _tensor_to_pil(pred_ema[i]).save(out_dir / f"{i:02d}_pred_ema.png")
        res = (pred_ema[i] - lr[i]).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1)
        res_vis = (res * scale).clamp(0, 1)
        _tensor_to_pil(res_vis).save(out_dir / f"{i:02d}_residual_amp.png")
    return out_dir


def compute_identity_psnr(val_loader: DataLoader, device: torch.device) -> float:
    total = 0.0
    n = 0
    with torch.no_grad():
        for lr, hr, _ in val_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            ps = tensor_psnr(lr, hr)
            total += float(ps.sum().item())
            n += int(lr.shape[0])
    return total / max(1, n)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    identity_psnr: float,
    use_ema_weights: bool,
    ema: ModelEMA | None,
) -> dict[str, float]:
    model.eval()
    if use_ema_weights and ema is not None:
        ema.apply_to(model)
    tot_psnr = 0.0
    tot_ratio = 0.0
    n = 0
    try:
        for lr, hr, _ in val_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            pred = model(lr).float()
            ps = tensor_psnr(pred, hr)
            tot_psnr += float(ps.sum().item())
            tot_ratio += residual_l1_ratio(pred, lr, hr) * lr.shape[0]
            n += lr.shape[0]
    finally:
        if use_ema_weights and ema is not None:
            ema.restore(model)
    mean_psnr = tot_psnr / max(1, n)
    mean_ratio = tot_ratio / max(1, n)
    return {
        "val_psnr": mean_psnr,
        "val_delta": mean_psnr - identity_psnr,
        "val_residual_ratio": mean_ratio,
    }


def _lr_at_epoch(
    epoch: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
    total_epochs: int,
) -> float:
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    if epoch >= total_epochs:
        return min_lr
    t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def run_training(
    config: TeacherResolvedConfig | dict[str, Any], *, project_root: Path | None = None, resume_path: Path | None = None
) -> dict[str, Any]:
    setup_logging()
    device = _resolve_device()
    if isinstance(config, TeacherResolvedConfig):
        cfg = config
    else:
        if project_root is None:
            raise ValueError("project_root is required when run_training is called with a raw config dict")
        cfg = resolve_teacher_config(config, project_root=project_root)
    if project_root is not None and cfg.project_root != project_root.resolve():
        cfg = replace(cfg, project_root=project_root.resolve())

    _set_seed(cfg.training.seed)

    run_root = cfg.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    validate_data_layout(cfg.training.data_root, cfg.project_root)
    train_pairs = collect_train_pairs(cfg.training.data_root, cfg.project_root)
    val_pairs = collect_val_pairs(cfg.training.data_root, cfg.project_root)
    if not train_pairs:
        raise RuntimeError("No training pairs found.")
    if not val_pairs:
        raise RuntimeError("No validation pairs found.")

    summary = summarize_pairs(train_pairs, val_pairs)
    logger.info("Train pairs: %d | Val pairs: %d", summary["train_pairs"], summary["val_pairs"])
    print(f"Paired train pairs: {summary['train_pairs']}")
    print(f"Paired val pairs: {summary['val_pairs']}")

    batch_size = _resolve_batch_size(
        cfg.training.batch_size,
        device,
        smoke_test=cfg.training.smoke_test_override,
    )
    epochs = int(cfg.training.epochs)
    base_lr = float(cfg.training.lr)
    min_lr = float(cfg.training.min_lr)
    weight_decay = float(cfg.training.weight_decay)
    warmup_epochs = int(cfg.training.warmup_epochs)
    grad_clip = float(cfg.training.grad_clip)
    train_workers = int(cfg.training.train_num_workers)
    val_workers = int(cfg.training.val_num_workers)
    log_step_interval = int(cfg.training.log_step_interval)

    model = build_teacher_model(cfg.architecture).to(device)

    loss_fn = TeacherCompositeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    ema = ModelEMA(model, cfg.training.ema_decay)
    use_amp = device.type == "cuda"
    amp_dtype = _autocast_dtype(device)
    try:
        from torch.amp import GradScaler as AmpGradScaler

        scaler = AmpGradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    except Exception:
        from torch.cuda.amp import GradScaler as CudaGradScaler

        scaler = CudaGradScaler(enabled=use_amp and amp_dtype == torch.float16)

    train_epoch_shared = Value("i", 0)
    train_loader, val_loader = make_dataloaders(
        train_pairs,
        val_pairs,
        batch_size=batch_size,
        patch_size=cfg.training.patch_size,
        seed=cfg.training.seed,
        train_workers=train_workers,
        val_workers=val_workers,
        device=device,
        train_epoch_counter=train_epoch_shared,
    )

    resolved_cfg = cfg.as_json()
    resolved_cfg["runtime"] = {"device": device.type, "batch_size_resolved": batch_size}

    global_step = 0
    start_epoch = 0
    best_val_psnr_ema = float("-inf")
    history_path = run_root / "history.json"
    metrics_path = run_root / "metrics.jsonl"
    checkpoint_teacher = None
    allow_legacy_checkpoint = False

    if resume_path is not None:
        ckpt = load_checkpoint(
            resume_path, model=model, optimizer=optimizer, ema_load=ema, map_location=device
        )
        checkpoint_teacher = checkpoint_teacher_metadata(ckpt)
        allow_legacy_checkpoint = is_legacy_teacher_checkpoint(ckpt)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_psnr_ema = float(ckpt.get("best_val_psnr_ema", float("-inf")))
        if allow_legacy_checkpoint:
            logger.warning(
                "Resuming from legacy teacher checkpoint %s without %s metadata; do not reuse its targets for distillation.",
                resume_path,
                cfg.teacher_model_version,
            )
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    t_run0 = time.time()
    meta_frozen = cfg.history_meta(hostname=socket.gethostname())
    _ = reconcile_run_state(
        run_root=run_root,
        history_meta=meta_frozen,
        resume_epoch=(start_epoch - 1) if resume_path is not None else None,
        allow_legacy_checkpoint=allow_legacy_checkpoint,
    )
    write_json(run_root / "run_config.json", resolved_cfg)

    identity_psnr = compute_identity_psnr(val_loader, device)
    logger.info("Validation LR identity PSNR (mean): %.4f dB", identity_psnr)
    print(f"LR identity baseline PSNR (val): {identity_psnr:.4f} dB")

    for epoch in range(start_epoch, epochs):
        train_epoch_shared.value = epoch
        lr_epoch = _lr_at_epoch(epoch, base_lr=base_lr, min_lr=min_lr, warmup_epochs=warmup_epochs, total_epochs=epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_epoch

        model.train()
        running_loss = 0.0
        running_parts: dict[str, float] = {"charbonnier": 0.0, "l1": 0.0, "edge": 0.0, "fft": 0.0}
        steps = 0
        ep_t0 = time.time()
        for batch_idx, (lr, hr, _) in enumerate(train_loader):
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            use_cast = use_amp and device.type == "cuda"
            if use_cast:
                try:
                    cast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
                except TypeError:
                    cast_ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                cast_ctx = nullcontext()
            with cast_ctx:
                pred = model(lr)
                loss, parts = loss_fn(pred, hr)

            if getattr(scaler, "is_enabled", lambda: False)():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            ema.update(model)
            global_step += 1
            running_loss += float(loss.detach())
            for k, v in parts.items():
                running_parts[k] += v
            steps += 1

            if log_step_interval > 0 and global_step % log_step_interval == 0:
                append_jsonl(
                    metrics_path,
                    {
                        "kind": "train_step",
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": running_loss / max(1, steps),
                        "lr": lr_epoch,
                        "elapsed_s": time.time() - t_run0,
                    },
                )

        train_loss = running_loss / max(1, steps)
        for k in running_parts:
            running_parts[k] /= max(1, steps)

        with torch.no_grad():
            model.eval()
            sample_lr, sample_hr = lr, hr
            sample_pred = model(sample_lr)
            res_rep = float(residual_supervision_l1(sample_pred, sample_lr, sample_hr).item())
            model.train()

        raw_metrics = evaluate_split(model, val_loader, device, identity_psnr, use_ema_weights=False, ema=None)
        ema_metrics = evaluate_split(model, val_loader, device, identity_psnr, use_ema_weights=True, ema=ema)
        is_new_best = ema_metrics["val_psnr"] > best_val_psnr_ema

        if is_new_best:
            best_val_psnr_ema = ema_metrics["val_psnr"]

        with torch.no_grad():
            vb = next(iter(val_loader))
            vlr, vhr = vb[0].to(device), vb[1].to(device)
            ema.apply_to(model)
            vpred = model(vlr)
            ema.restore(model)
            sample_dir = _save_val_samples(run_root, epoch, vlr, vhr, vpred)

        epoch_record = {
            "kind": "epoch",
            "epoch": epoch,
            "elapsed_s": time.time() - t_run0,
            "epoch_wall_s": time.time() - ep_t0,
            "train_loss": train_loss,
            "train_loss_components": running_parts,
            "residual_report_l1": res_rep,
            "lr": lr_epoch,
            "val_psnr_raw": raw_metrics["val_psnr"],
            "val_psnr_ema": ema_metrics["val_psnr"],
            "val_delta_raw": raw_metrics["val_delta"],
            "val_delta_ema": ema_metrics["val_delta"],
            "val_residual_ratio_raw": raw_metrics["val_residual_ratio"],
            "val_residual_ratio_ema": ema_metrics["val_residual_ratio"],
            "best_val_psnr_ema": best_val_psnr_ema,
            "identity_psnr": identity_psnr,
            "global_step": global_step,
        }
        existing_metrics = load_jsonl(metrics_path)
        existing_epoch_rows = [rec for rec in existing_metrics if rec.get("kind") == "epoch"]
        if existing_epoch_rows and int(existing_epoch_rows[-1]["epoch"]) == epoch:
            raise RuntimeError(f"Metrics already contain epoch {epoch}; reconciliation should have trimmed it before training.")
        append_jsonl(metrics_path, epoch_record)

        latest = {
            "epoch": epoch,
            "teacher_model_version": cfg.teacher_model_version,
            "config_fingerprint": cfg.config_fingerprint,
            "identity_psnr": identity_psnr,
            "val_psnr_raw": raw_metrics["val_psnr"],
            "val_psnr_ema": ema_metrics["val_psnr"],
            "val_delta_raw": raw_metrics["val_delta"],
            "val_delta_ema": ema_metrics["val_delta"],
            "best_val_psnr_ema": best_val_psnr_ema,
            "train_loss": train_loss,
            "checkpoint_best": str(run_root / "checkpoints" / "best_ema.pth"),
            "checkpoint_latest": str(run_root / "checkpoints" / "latest.pth"),
            "val_samples_dir": str(sample_dir),
            "updated_at": wall_ts(),
        }
        prev_hist = load_history(history_path)
        hist_payload = merge_history_meta(prev_hist, meta_frozen, [epoch_record])
        atomic_write_json(history_path, hist_payload)
        write_json(run_root / "latest_status.json", latest)

        checkpoint_extra = {
            "identity_psnr": identity_psnr,
            "architecture": cfg.architecture.as_json(),
            "profile": cfg.active_profile,
        }
        if checkpoint_teacher:
            checkpoint_extra["resumed_from_teacher_metadata"] = checkpoint_teacher
        if is_new_best:
            save_checkpoint(
                run_root / "checkpoints" / "best_ema.pth",
                model=model,
                optimizer=optimizer,
                scheduler_state=None,
                epoch=epoch,
                global_step=global_step,
                best_val_psnr_ema=best_val_psnr_ema,
                ema_state=ema.state_dict(),
                teacher_metadata=cfg.checkpoint_metadata(),
                extra=checkpoint_extra,
            )
            logger.info("New best EMA val PSNR: %.4f", best_val_psnr_ema)

        save_checkpoint(
            run_root / "checkpoints" / "latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler_state=None,
            epoch=epoch,
            global_step=global_step,
            best_val_psnr_ema=best_val_psnr_ema,
            ema_state=ema.state_dict(),
            teacher_metadata=cfg.checkpoint_metadata(),
            extra=checkpoint_extra,
        )

        logger.info(
            "Epoch %d/%d | train_loss=%.5f | val_psnr raw=%.3f ema=%.3f | delta_ema=%.3f | best_ema=%.3f | samples=%s",
            epoch + 1,
            epochs,
            train_loss,
            raw_metrics["val_psnr"],
            ema_metrics["val_psnr"],
            ema_metrics["val_delta"],
            best_val_psnr_ema,
            sample_dir,
        )
        print(
            f"Epoch {epoch+1}/{epochs} train_loss={train_loss:.5f} "
            f"val_psnr raw={raw_metrics['val_psnr']:.3f} ema={ema_metrics['val_psnr']:.3f} "
            f"delta_ema={ema_metrics['val_delta']:.3f} best_ema={best_val_psnr_ema:.3f} "
            f"identity={identity_psnr:.3f}"
        )

    return {
        "run_root": str(run_root),
        "best_val_psnr_ema": best_val_psnr_ema,
        "identity_psnr": identity_psnr,
        "teacher_model_version": cfg.teacher_model_version,
        "config_fingerprint": cfg.config_fingerprint,
    }
