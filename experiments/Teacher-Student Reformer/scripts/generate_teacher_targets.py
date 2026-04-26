#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tensor01_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.detach().float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_save_tensor(path: Path, t: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _tensor01_to_pil(t).save(path)


def _load_cfg_from_yaml(cfg_path: Path, project_root: Path) -> "TeacherResolvedConfig":
    from restormer_teacher.config import resolve_teacher_config

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw.setdefault("training", {})["config_path"] = str(cfg_path)
    return resolve_teacher_config(raw, project_root=project_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate teacher predictions + metadata for distillation.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best_ema.pth (or latest with EMA)")
    p.add_argument("--config", type=str, default="", help="YAML if checkpoint lacks architecture extra")
    p.add_argument("--profile", type=str, default="large", help="Profile key when using --config fallback")
    p.add_argument("--data-root", type=str, default="Data", help="Lab3 Data root")
    p.add_argument("--output-dir", type=str, required=True, help="e.g. runs/restormer_teacher/<run_id>/teacher_targets")
    p.add_argument("--only-save-improved", action="store_true", help="Skip PNG/npy when teacher does not beat identity")
    p.add_argument("--save-residuals", action="store_true", help="Save teacher_pred - lr as .npy (and amplified PNG)")
    p.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Allow target generation from pre-fix / unversioned checkpoints for inspection only.",
    )
    p.add_argument("--max-images", type=int, default=0, help="If >0, only process first N pairs (debug)")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    pkg = _pkg_root()
    project_root = _project_root()
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

    from restormer_teacher.checkpointing import is_legacy_teacher_checkpoint
    from restormer_teacher.config import TeacherResolvedConfig
    from restormer_teacher.data import collect_train_pairs, pil_rgb, pil_to_tensor_chw01
    from restormer_teacher.ema import ModelEMA
    from restormer_teacher.metrics import tensor_psnr
    from restormer_teacher.model import build_teacher_model

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    is_legacy = is_legacy_teacher_checkpoint(ckpt)
    if is_legacy and not args.allow_legacy_checkpoint:
        raise SystemExit(
            "Checkpoint is legacy / pre-fix and is rejected for target generation by default. "
            "Retrain the teacher, or pass --allow-legacy-checkpoint for inspection-only output."
        )

    extra = ckpt.get("extra") or {}
    teacher_meta = ckpt.get("teacher") or {}
    cfg: TeacherResolvedConfig | None = None
    arch = teacher_meta.get("architecture") or extra.get("architecture")
    if arch is None:
        if not args.config:
            raise SystemExit("Checkpoint missing extra.architecture; pass --config")
        cfg = _load_cfg_from_yaml(Path(args.config).resolve(), project_root)
        if cfg.active_profile != args.profile and args.profile:
            raw = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
            raw["active_profile"] = args.profile
            raw.setdefault("training", {})["config_path"] = str(Path(args.config).resolve())
            from restormer_teacher.config import resolve_teacher_config

            cfg = resolve_teacher_config(raw, project_root=project_root)
        arch = cfg.architecture

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_teacher_model(arch).to(device)
    model.load_state_dict(ckpt["model"])
    if "ema" in ckpt:
        ema = ModelEMA(model, float(ckpt["ema"].get("decay", 0.999)))
        ema.load_state_dict(ckpt["ema"])
        ema.apply_to(model)
    model.eval()

    data_root = Path(args.data_root).expanduser()
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()

    pairs = collect_train_pairs(data_root, project_root)
    if args.max_images and args.max_images > 0:
        pairs = pairs[: args.max_images]

    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    pred_dir = out_dir / "predictions"
    npy_dir = out_dir / "npy"
    res_dir = out_dir / "residuals_npy"
    meta_path = out_dir / "metadata.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("", encoding="utf-8")

    for lr_path, hr_path, pair_name in pairs:
        stem = lr_path.stem
        lr_img = pil_rgb(lr_path)
        hr_img = pil_rgb(hr_path)
        if lr_img.size != (256, 256) or hr_img.size != (256, 256):
            raise ValueError(
                f"Expected 256x256 for pair {pair_name!r} (LR={lr_path}), "
                f"got LR{lr_img.size} HR{hr_img.size}"
            )
        lr_t = pil_to_tensor_chw01(lr_img).unsqueeze(0).to(device)
        hr_t = pil_to_tensor_chw01(hr_img).unsqueeze(0).to(device)
        pred = model(lr_t).float()
        id_psnr = float(tensor_psnr(lr_t, hr_t).mean().item())
        te_psnr = float(tensor_psnr(pred, hr_t).mean().item())
        delta = te_psnr - id_psnr
        improved = te_psnr > id_psnr
        pred_png = pred_dir / f"{stem}.png"
        record = {
            "basename": stem,
            "teacher_model_version": teacher_meta.get("teacher_model_version", "legacy"),
            "config_fingerprint": teacher_meta.get("config_fingerprint", ""),
            "lr_path": str(lr_path),
            "hr_path": str(hr_path),
            "teacher_out_path": str(pred_png),
            "identity_psnr": id_psnr,
            "teacher_psnr": te_psnr,
            "teacher_delta_psnr": delta,
            "use_for_distillation": improved,
        }

        save_files = not args.only_save_improved or improved
        if save_files:
            pred_dir.mkdir(parents=True, exist_ok=True)
            _pil_save_tensor(pred_png, pred[0])

        if save_files and args.save_residuals:
            res = pred - lr_t
            npy_dir.mkdir(parents=True, exist_ok=True)
            res_dir.mkdir(parents=True, exist_ok=True)
            np.save(npy_dir / f"{stem}_pred.npy", pred[0].float().cpu().numpy())
            np.save(res_dir / f"{stem}_teacher_residual.npy", res[0].float().cpu().numpy())
            amp = (res[0].abs() * 8.0).clamp(0, 1)
            _pil_save_tensor(res_dir / f"{stem}_teacher_residual_amp.png", amp)

        with meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"Wrote metadata to {meta_path} ({len(pairs)} lines)")


if __name__ == "__main__":
    main()
