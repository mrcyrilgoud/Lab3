from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from restormer_teacher.config import TEACHER_MODEL_VERSION


def checkpoint_teacher_metadata(ckpt: dict[str, Any]) -> dict[str, Any]:
    teacher = ckpt.get("teacher")
    return dict(teacher) if isinstance(teacher, dict) else {}


def is_legacy_teacher_checkpoint(ckpt: dict[str, Any]) -> bool:
    teacher = checkpoint_teacher_metadata(ckpt)
    return teacher.get("teacher_model_version") != TEACHER_MODEL_VERSION


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict[str, Any] | None,
    epoch: int,
    global_step: int,
    best_val_psnr_ema: float,
    ema_state: dict[str, Any],
    teacher_metadata: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "best_val_psnr_ema": best_val_psnr_ema,
        "ema": ema_state,
        "teacher": dict(teacher_metadata),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    ema_load: Any,
    map_location: torch.device | None = None,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema_load is not None and "ema" in ckpt:
        ema_load.load_state_dict(ckpt["ema"], map_location=map_location)
    return ckpt
