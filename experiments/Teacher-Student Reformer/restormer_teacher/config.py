from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


TEACHER_MODEL_VERSION = "restormer_teacher_v2"


def _normalize_path(project_root: Path, path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (project_root / candidate).resolve()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TeacherArchitectureConfig:
    dim: int
    num_blocks: tuple[int, int, int, int]
    num_refinement_blocks: int
    heads: tuple[int, int, int, int]
    ffn_expansion_factor: float

    def as_json(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "num_blocks": list(self.num_blocks),
            "num_refinement_blocks": self.num_refinement_blocks,
            "heads": list(self.heads),
            "ffn_expansion_factor": self.ffn_expansion_factor,
        }


@dataclass(frozen=True)
class TeacherTrainingConfig:
    data_root: Path
    artifact_root: Path
    run_name: str
    started_day: str
    epochs: int
    batch_size: int | str | None
    lr: float
    min_lr: float
    weight_decay: float
    warmup_epochs: int
    patch_size: int
    ema_decay: float
    grad_clip: float
    seed: int
    train_num_workers: int
    val_num_workers: int
    log_step_interval: int
    smoke_test_override: bool = False
    modal_app: str = ""
    config_path: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "artifact_root": str(self.artifact_root),
            "run_name": self.run_name,
            "started_day": self.started_day,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "min_lr": self.min_lr,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "patch_size": self.patch_size,
            "ema_decay": self.ema_decay,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "train_num_workers": self.train_num_workers,
            "val_num_workers": self.val_num_workers,
            "log_step_interval": self.log_step_interval,
            "smoke_test_override": self.smoke_test_override,
            "modal_app": self.modal_app,
            "config_path": self.config_path,
        }


@dataclass(frozen=True)
class TeacherResolvedConfig:
    project_root: Path
    teacher_model_version: str
    active_profile: str
    config_path: str
    config_fingerprint: str
    architecture: TeacherArchitectureConfig
    training: TeacherTrainingConfig

    @property
    def run_root(self) -> Path:
        return self.training.artifact_root / self.training.started_day / self.training.run_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_root / "checkpoints"

    def as_json(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "teacher_model_version": self.teacher_model_version,
            "active_profile": self.active_profile,
            "config_path": self.config_path,
            "config_fingerprint": self.config_fingerprint,
            "architecture": self.architecture.as_json(),
            "training": self.training.as_json(),
        }

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "teacher_model_version": self.teacher_model_version,
            "active_profile": self.active_profile,
            "architecture": self.architecture.as_json(),
            "config_path": self.config_path,
            "config_fingerprint": self.config_fingerprint,
            "run_name": self.training.run_name,
            "started_day": self.training.started_day,
        }

    def history_meta(self, *, hostname: str) -> dict[str, Any]:
        return {
            "run_id": self.training.run_name,
            "teacher_model_version": self.teacher_model_version,
            "config_path": self.config_path,
            "config_fingerprint": self.config_fingerprint,
            "profile": self.active_profile,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "started_day": self.training.started_day,
            "data_root": str(self.training.data_root),
            "artifact_root": str(self.training.artifact_root),
            "hostname": hostname,
            "modal_app": self.training.modal_app,
            "architecture": self.architecture.as_json(),
        }

    def with_training_updates(self, **updates: Any) -> TeacherResolvedConfig:
        return replace(self, training=replace(self.training, **updates))


def teacher_config_from_json(payload: dict[str, Any]) -> TeacherResolvedConfig:
    arch_payload = dict(payload["architecture"])
    architecture = TeacherArchitectureConfig(
        dim=int(arch_payload["dim"]),
        num_blocks=tuple(int(x) for x in arch_payload["num_blocks"]),
        num_refinement_blocks=int(arch_payload["num_refinement_blocks"]),
        heads=tuple(int(x) for x in arch_payload["heads"]),
        ffn_expansion_factor=float(arch_payload["ffn_expansion_factor"]),
    )
    tr_payload = dict(payload["training"])
    training = TeacherTrainingConfig(
        data_root=Path(tr_payload["data_root"]).expanduser(),
        artifact_root=Path(tr_payload["artifact_root"]).expanduser(),
        run_name=str(tr_payload["run_name"]),
        started_day=str(tr_payload["started_day"]),
        epochs=int(tr_payload["epochs"]),
        batch_size=tr_payload.get("batch_size", "auto"),
        lr=float(tr_payload["lr"]),
        min_lr=float(tr_payload["min_lr"]),
        weight_decay=float(tr_payload["weight_decay"]),
        warmup_epochs=int(tr_payload["warmup_epochs"]),
        patch_size=int(tr_payload["patch_size"]),
        ema_decay=float(tr_payload["ema_decay"]),
        grad_clip=float(tr_payload["grad_clip"]),
        seed=int(tr_payload["seed"]),
        train_num_workers=int(tr_payload["train_num_workers"]),
        val_num_workers=int(tr_payload["val_num_workers"]),
        log_step_interval=int(tr_payload.get("log_step_interval", 0) or 0),
        smoke_test_override=bool(tr_payload.get("smoke_test_override", False)),
        modal_app=str(tr_payload.get("modal_app", "")),
        config_path=str(tr_payload.get("config_path", payload.get("config_path", ""))),
    )
    return TeacherResolvedConfig(
        project_root=Path(payload["project_root"]).expanduser(),
        teacher_model_version=str(payload.get("teacher_model_version", TEACHER_MODEL_VERSION)),
        active_profile=str(payload["active_profile"]),
        config_path=str(payload.get("config_path", "")),
        config_fingerprint=str(payload["config_fingerprint"]),
        architecture=architecture,
        training=training,
    )


def resolve_teacher_config(raw_cfg: dict[str, Any], *, project_root: Path) -> TeacherResolvedConfig:
    profiles = raw_cfg["profiles"]
    profile_name = str(raw_cfg.get("active_profile", "large"))
    tr = dict(raw_cfg["training"])
    if bool(tr.get("smoke_test_override", False)):
        profile_name = "smoke"
        tr["epochs"] = 1
    arch_raw = dict(profiles[profile_name])
    architecture = TeacherArchitectureConfig(
        dim=int(arch_raw["dim"]),
        num_blocks=tuple(int(x) for x in arch_raw["num_blocks"]),
        num_refinement_blocks=int(arch_raw["num_refinement_blocks"]),
        heads=tuple(int(x) for x in arch_raw["heads"]),
        ffn_expansion_factor=float(arch_raw["ffn_expansion_factor"]),
    )
    training = TeacherTrainingConfig(
        data_root=_normalize_path(project_root, tr.get("data_root", "Data")),
        artifact_root=_normalize_path(project_root, tr.get("artifact_root", project_root / "runs")),
        run_name=str(tr["run_name"]),
        started_day=str(tr.get("started_day") or time.strftime("%Y-%m-%d", time.gmtime())),
        epochs=int(tr.get("epochs", 100)),
        batch_size=tr.get("batch_size", "auto"),
        lr=float(tr.get("lr", 2e-4)),
        min_lr=float(tr.get("min_lr", 1e-6)),
        weight_decay=float(tr.get("weight_decay", 1e-4)),
        warmup_epochs=int(tr.get("warmup_epochs", 3)),
        patch_size=int(tr.get("patch_size", 192)),
        ema_decay=float(tr.get("ema_decay", 0.999)),
        grad_clip=float(tr.get("grad_clip", 1.0)),
        seed=int(tr.get("seed", 42)),
        train_num_workers=int(tr.get("train_num_workers", 4)),
        val_num_workers=int(tr.get("val_num_workers", 2)),
        log_step_interval=int(tr.get("log_step_interval", 0) or 0),
        smoke_test_override=bool(tr.get("smoke_test_override", False)),
        modal_app=str(tr.get("modal_app", "")),
        config_path=str(tr.get("config_path", "")),
    )
    candidate = {
        "teacher_model_version": TEACHER_MODEL_VERSION,
        "active_profile": profile_name,
        "config_path": training.config_path,
        "architecture": architecture.as_json(),
        "training": training.as_json(),
    }
    return TeacherResolvedConfig(
        project_root=project_root.resolve(),
        teacher_model_version=TEACHER_MODEL_VERSION,
        active_profile=profile_name,
        config_path=training.config_path,
        config_fingerprint=_fingerprint(candidate),
        architecture=architecture,
        training=training,
    )
