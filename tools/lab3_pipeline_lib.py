from __future__ import annotations

import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

try:
    import onnx
except Exception:
    onnx = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
RESAMPLING_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
SAFE_LEAF_MODULES = ("Conv2d", "LeakyReLU")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    notebook_slug: str
    channels: int
    num_blocks: int
    architecture: str = "wide_residual"
    slope: float = 0.10
    groups: int = 1
    bottleneck_ratio: float = 0.5
    stem_kernel_size: int = 3
    body_kernel_size: int = 3
    alt_kernel_size: int = 3
    stages: int = 1
    search_tier: str = "bounded"
    notes: str = ""

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


CANDIDATE_REGISTRY: dict[str, CandidateSpec] = {
    "wide_residual_nobn_v1": CandidateSpec(
        candidate_id="wide_residual_nobn_v1",
        notebook_slug="wide_residual_nobn",
        channels=64,
        num_blocks=16,
        search_tier="bounded",
        notes="Baseline NPU-first no-BN residual CNN.",
    ),
    "wide_residual_nobn_compact": CandidateSpec(
        candidate_id="wide_residual_nobn_compact",
        notebook_slug="wide_residual_nobn_compact",
        channels=48,
        num_blocks=12,
        search_tier="bounded",
        notes="Smaller baseline for latency pressure.",
    ),
    "wide_residual_nobn_deeper": CandidateSpec(
        candidate_id="wide_residual_nobn_deeper",
        notebook_slug="wide_residual_nobn_deeper",
        channels=64,
        num_blocks=20,
        search_tier="bounded",
        notes="Depth increase inside the same safe operator family.",
    ),
    "wide_residual_nobn_wider": CandidateSpec(
        candidate_id="wide_residual_nobn_wider",
        notebook_slug="wide_residual_nobn_wider",
        channels=80,
        num_blocks=16,
        search_tier="bounded",
        notes="Width increase inside the same safe operator family.",
    ),
    "wide_residual_nobn_xwide_deep": CandidateSpec(
        candidate_id="wide_residual_nobn_xwide_deep",
        notebook_slug="wide_residual_nobn_xwide_deep",
        channels=80,
        num_blocks=20,
        search_tier="extreme",
        notes="Explicitly whitelisted wider/deeper same-family variant.",
    ),
    "mixed_kernel_residual_nobn_compact": CandidateSpec(
        candidate_id="mixed_kernel_residual_nobn_compact",
        notebook_slug="mixed_kernel_residual_nobn_compact",
        architecture="mixed_kernel_residual",
        channels=48,
        num_blocks=12,
        body_kernel_size=3,
        alt_kernel_size=5,
        search_tier="bounded",
        notes="Alternates 3x3 and 5x5 residual kernels while staying in Conv2d/LeakyReLU.",
    ),
    "bottleneck_residual_nobn_mid": CandidateSpec(
        candidate_id="bottleneck_residual_nobn_mid",
        notebook_slug="bottleneck_residual_nobn_mid",
        architecture="bottleneck_residual",
        channels=64,
        num_blocks=16,
        bottleneck_ratio=0.5,
        search_tier="bounded",
        notes="1x1/3x3/1x1 bottleneck residual blocks for a different capacity layout.",
    ),
    "grouped_residual_nobn_g4": CandidateSpec(
        candidate_id="grouped_residual_nobn_g4",
        notebook_slug="grouped_residual_nobn_g4",
        architecture="grouped_residual",
        channels=64,
        num_blocks=16,
        groups=4,
        search_tier="bounded",
        notes="Uses grouped 3x3 body convolutions to test cheaper mixing inside the same operator family.",
    ),
    "two_stage_residual_nobn_compact": CandidateSpec(
        candidate_id="two_stage_residual_nobn_compact",
        notebook_slug="two_stage_residual_nobn_compact",
        architecture="two_stage_residual",
        channels=40,
        num_blocks=6,
        stages=2,
        search_tier="bounded",
        notes="Two smaller same-resolution residual refiners chained in series.",
    ),
}


@dataclass
class PipelineConfig:
    project_root: Path
    data_root: Path
    run_name: str
    artifact_root: Path | None = None
    candidate_id: str = "wide_residual_nobn_v1"
    seed: int = 255
    batch_size: int = 24
    num_epochs: int = 80
    learning_rate: float = 3e-4
    weight_decay: float = 2e-4
    train_patch_size: int = 224
    eval_size: int = 256
    train_num_workers: int = 0
    eval_num_workers: int = 0
    warmup_epochs: int = 5
    ema_decay: float = 0.999
    use_amp: bool = True
    channels_last: bool = True
    calibration_count: int = 128
    run_training: bool = True
    run_onnx_export: bool = True
    verify_onnx_export: bool = True
    run_mxq_compile: bool = False
    compile_command_template: str = ""
    compile_extra_args: list[str] = field(default_factory=list)
    train_pair_limit: int | None = None
    val_pair_limit: int | None = None
    material_regression_tolerance: float = 0.05
    backend: str = "local"
    started_day: str = ""
    modal_app_name: str = ""
    modal_function_name: str = ""
    modal_gpu: str = ""
    modal_timeout_minutes: int = 0
    modal_data_volume: str = ""
    modal_runs_volume: str = ""
    synced_local_run_root: str = ""

    def candidate(self) -> CandidateSpec:
        return get_candidate_spec(self.candidate_id)

    def resolved_artifact_root(self) -> Path:
        return normalize_path(self.artifact_root or self.project_root)

    def resolved_started_day(self) -> str:
        return self.started_day or time.strftime("%Y-%m-%d")

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["data_root"] = str(self.data_root)
        payload["artifact_root"] = str(self.resolved_artifact_root())
        payload["started_day"] = self.resolved_started_day()
        payload["candidate"] = self.candidate().as_json()
        return payload


@dataclass
class RunLayout:
    artifact_root: Path
    runs_root: Path
    day_root: Path
    run_day: str
    run_root: Path
    checkpoint_dir: Path
    export_dir: Path
    calibration_dir: Path
    notebook_dir: Path
    status_path: Path
    metrics_path: Path
    summary_path: Path
    config_path: Path
    report_path: Path
    notebook_path: Path
    mxq_path: Path


def get_candidate_spec(candidate_id: str) -> CandidateSpec:
    if candidate_id not in CANDIDATE_REGISTRY:
        raise KeyError(f"Unknown candidate_id={candidate_id}; choose from {sorted(CANDIDATE_REGISTRY)}")
    return CANDIDATE_REGISTRY[candidate_id]


def list_candidate_ids(include_extreme: bool = True) -> list[str]:
    return [
        candidate_id
        for candidate_id, spec in CANDIDATE_REGISTRY.items()
        if include_extreme or spec.search_tier != "extreme"
    ]


def build_run_layout(artifact_root: Path, run_name: str, notebook_slug: str, run_day: str | None = None) -> RunLayout:
    artifact_root = normalize_path(artifact_root)
    run_day = run_day or time.strftime("%Y-%m-%d")
    day_root = artifact_root / "runs" / run_day
    run_root = day_root / run_name
    checkpoint_dir = run_root / "checkpoints"
    export_dir = run_root / "exports"
    calibration_dir = export_dir / "calibration"
    notebook_dir = run_root / "notebooks"
    for path in [day_root, run_root, checkpoint_dir, export_dir, calibration_dir, notebook_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return RunLayout(
        artifact_root=artifact_root,
        runs_root=artifact_root / "runs",
        day_root=day_root,
        run_day=run_day,
        run_root=run_root,
        checkpoint_dir=checkpoint_dir,
        export_dir=export_dir,
        calibration_dir=calibration_dir,
        notebook_dir=notebook_dir,
        status_path=run_root / "latest_status.json",
        metrics_path=run_root / "metrics.jsonl",
        summary_path=run_root / "summary.json",
        config_path=run_root / "run_config.json",
        report_path=run_root / "report.json",
        notebook_path=notebook_dir / f"lab3_{notebook_slug}_autopilot.ipynb",
        mxq_path=export_dir / "best.mxq",
    )


def run_layout_from_config(cfg: PipelineConfig) -> RunLayout:
    return build_run_layout(
        cfg.resolved_artifact_root(),
        cfg.run_name,
        cfg.candidate().notebook_slug,
        run_day=cfg.resolved_started_day(),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        with nullcontext():
            yield


def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def tensor_psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((a.float().clamp(0.0, 1.0) - b.float().clamp(0.0, 1.0)) ** 2, dim=(-3, -2, -1))
    mse = mse.clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def normalize_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def pipeline_config_from_json(payload: dict[str, Any]) -> PipelineConfig:
    values = dict(payload)
    for key in ["project_root", "data_root", "artifact_root"]:
        if values.get(key):
            values[key] = normalize_path(values[key])
    allowed = {field.name for field in fields(PipelineConfig)}
    filtered = {key: value for key, value in values.items() if key in allowed}
    return PipelineConfig(**filtered)


def collect_paired_by_subfolder(lr_root: Path, hr_root: Path) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    if not lr_root.exists() or not hr_root.exists():
        return pairs
    for hr_dir in sorted(path for path in hr_root.iterdir() if path.is_dir()):
        suffix = hr_dir.name.replace("HR_train", "")
        lr_dir = lr_root / f"LR_train{suffix}"
        if not lr_dir.exists():
            continue
        hr_images = {path.stem: path for path in sorted(hr_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES}
        lr_images = {path.stem: path for path in sorted(lr_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES}
        for stem in sorted(set(hr_images) & set(lr_images)):
            pairs.append((lr_images[stem], hr_images[stem], f"{hr_dir.name}/{stem}"))
    return pairs


def collect_paired_flat(lr_dir: Path, hr_dir: Path) -> list[tuple[Path, Path, str]]:
    if not lr_dir.exists() or not hr_dir.exists():
        return []
    hr_images = {path.stem: path for path in sorted(hr_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES}
    lr_images = {path.stem: path for path in sorted(lr_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES}
    return [(lr_images[stem], hr_images[stem], stem) for stem in sorted(set(hr_images) & set(lr_images))]


def limit_pairs(
    pairs: list[tuple[Path, Path, str]],
    limit: int | None,
) -> list[tuple[Path, Path, str]]:
    if limit is None or limit <= 0 or len(pairs) <= limit:
        return pairs
    return pairs[:limit]


def collect_train_pairs(data_root: Path, limit: int | None = None) -> list[tuple[Path, Path, str]]:
    structured = collect_paired_by_subfolder(data_root / "LR_train", data_root / "HR_train")
    pairs = structured if structured else collect_paired_flat(data_root / "train" / "LR", data_root / "train" / "HR")
    return limit_pairs(pairs, limit)


def collect_val_pairs(data_root: Path, limit: int | None = None) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    for lr_dir, hr_dir in [
        (data_root / "LR_val", data_root / "HR_val"),
        (data_root / "val" / "LR_val", data_root / "val" / "HR_val"),
        (data_root / "val" / "LR", data_root / "val" / "HR"),
    ]:
        pairs = collect_paired_flat(lr_dir, hr_dir)
        if pairs:
            break
    return limit_pairs(pairs, limit)


def summarize_pairs(train_pairs: list[tuple[Path, Path, str]], val_pairs: list[tuple[Path, Path, str]]) -> dict[str, Any]:
    return {
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_preview": [name for _, _, name in train_pairs[:3]],
        "val_preview": [name for _, _, name in val_pairs[:3]],
    }


def random_crop_pair(lr_img: Image.Image, hr_img: Image.Image, size: int, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    width, height = lr_img.size
    if min(width, height) < size:
        lr_img = ImageOps.fit(lr_img, (size, size), method=RESAMPLING_BICUBIC)
        hr_img = ImageOps.fit(hr_img, (size, size), method=RESAMPLING_BICUBIC)
        return lr_img, hr_img
    left = rng.randint(0, width - size)
    top = rng.randint(0, height - size)
    box = (left, top, left + size, top + size)
    return lr_img.crop(box), hr_img.crop(box)


def augment_pair(lr_img: Image.Image, hr_img: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    if rng.random() < 0.5:
        lr_img = ImageOps.mirror(lr_img)
        hr_img = ImageOps.mirror(hr_img)
    if rng.random() < 0.5:
        lr_img = ImageOps.flip(lr_img)
        hr_img = ImageOps.flip(hr_img)
    rotations = rng.randint(0, 3)
    if rotations:
        angle = 90 * rotations
        lr_img = lr_img.rotate(angle)
        hr_img = hr_img.rotate(angle)
    return lr_img, hr_img


class PairedSRDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path, str]], train: bool, seed: int, patch_size: int, eval_size: int):
        self.pairs = pairs
        self.train = train
        self.seed = seed
        self.patch_size = patch_size
        self.eval_size = eval_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        lr_path, hr_path, name = self.pairs[index]
        lr_img = pil_rgb(lr_path)
        hr_img = pil_rgb(hr_path)
        rng = random.Random(self.seed + index)
        if self.train:
            lr_img, hr_img = random_crop_pair(lr_img, hr_img, self.patch_size, rng)
            lr_img, hr_img = augment_pair(lr_img, hr_img, rng)
        else:
            if lr_img.size != (self.eval_size, self.eval_size):
                lr_img = ImageOps.fit(lr_img, (self.eval_size, self.eval_size), method=RESAMPLING_BICUBIC)
            if hr_img.size != (self.eval_size, self.eval_size):
                hr_img = ImageOps.fit(hr_img, (self.eval_size, self.eval_size), method=RESAMPLING_BICUBIC)
        return pil_to_tensor(lr_img), pil_to_tensor(hr_img), name


def make_dataloaders(
    train_pairs: list[tuple[Path, Path, str]],
    val_pairs: list[tuple[Path, Path, str]],
    cfg: PipelineConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        PairedSRDataset(train_pairs, train=True, seed=cfg.seed, patch_size=cfg.train_patch_size, eval_size=cfg.eval_size),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.train_num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        PairedSRDataset(val_pairs, train=False, seed=cfg.seed, patch_size=cfg.train_patch_size, eval_size=cfg.eval_size),
        batch_size=max(1, min(cfg.batch_size, 8)),
        shuffle=False,
        num_workers=cfg.eval_num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def make_activation(slope: float) -> nn.Module:
    return nn.LeakyReLU(negative_slope=slope, inplace=True)


def init_tail_small(layer: nn.Conv2d, scale: float = 1e-3) -> None:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
    layer.weight.data.mul_(scale)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class WideResidualBlock(nn.Module):
    def __init__(self, channels: int, slope: float, kernel_size: int = 3, groups: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=groups, bias=True)
        self.act = make_activation(slope)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=groups, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class MixedKernelResidualBlock(nn.Module):
    def __init__(self, channels: int, slope: float, kernel_size: int, alt_kernel_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=True),
            make_activation(slope),
            nn.Conv2d(channels, channels, alt_kernel_size, padding=alt_kernel_size // 2, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, channels: int, slope: float, bottleneck_ratio: float):
        super().__init__()
        hidden = max(8, int(round(channels * bottleneck_ratio)))
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, padding=0, bias=True),
            make_activation(slope),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=True),
            make_activation(slope),
            nn.Conv2d(hidden, channels, 1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class WideResidualNoBNSR(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        num_blocks: int = 16,
        slope: float = 0.10,
        body_kernel_size: int = 3,
        groups: int = 1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=True),
            make_activation(slope),
        )
        self.body = nn.Sequential(
            *[WideResidualBlock(channels, slope, kernel_size=body_kernel_size, groups=groups) for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1, bias=True)
        init_tail_small(self.tail)

    def predict_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.body(self.stem(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.predict_delta(x)


class MixedKernelResidualNoBNSR(nn.Module):
    def __init__(self, channels: int, num_blocks: int, slope: float, kernel_size: int, alt_kernel_size: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=True),
            make_activation(slope),
        )
        blocks = []
        for index in range(num_blocks):
            first_kernel = kernel_size if index % 2 == 0 else alt_kernel_size
            second_kernel = alt_kernel_size if index % 2 == 0 else kernel_size
            blocks.append(MixedKernelResidualBlock(channels, slope, first_kernel, second_kernel))
        self.body = nn.Sequential(*blocks)
        self.tail = nn.Conv2d(channels, 3, 3, padding=1, bias=True)
        init_tail_small(self.tail)

    def predict_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.body(self.stem(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.predict_delta(x)


class BottleneckResidualNoBNSR(nn.Module):
    def __init__(self, channels: int, num_blocks: int, slope: float, bottleneck_ratio: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=True),
            make_activation(slope),
        )
        self.body = nn.Sequential(
            *[BottleneckResidualBlock(channels, slope, bottleneck_ratio=bottleneck_ratio) for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1, bias=True)
        init_tail_small(self.tail)

    def predict_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.body(self.stem(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.predict_delta(x)


class TwoStageResidualNoBNSR(nn.Module):
    def __init__(self, channels: int, num_blocks: int, slope: float, stages: int):
        super().__init__()
        self.stages = nn.ModuleList(
            [WideResidualNoBNSR(channels=channels, num_blocks=num_blocks, slope=slope) for _ in range(stages)]
        )

    def predict_delta(self, x: torch.Tensor) -> torch.Tensor:
        current = x
        for stage in self.stages:
            current = stage(current)
        return current - x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = x
        for stage in self.stages:
            current = stage(current)
        return current


def build_model(cfg: PipelineConfig) -> nn.Module:
    candidate = cfg.candidate()
    if candidate.architecture == "wide_residual":
        return WideResidualNoBNSR(
            channels=candidate.channels,
            num_blocks=candidate.num_blocks,
            slope=candidate.slope,
            body_kernel_size=candidate.body_kernel_size,
            groups=candidate.groups,
        )
    if candidate.architecture == "mixed_kernel_residual":
        return MixedKernelResidualNoBNSR(
            channels=candidate.channels,
            num_blocks=candidate.num_blocks,
            slope=candidate.slope,
            kernel_size=candidate.body_kernel_size,
            alt_kernel_size=candidate.alt_kernel_size,
        )
    if candidate.architecture == "bottleneck_residual":
        return BottleneckResidualNoBNSR(
            channels=candidate.channels,
            num_blocks=candidate.num_blocks,
            slope=candidate.slope,
            bottleneck_ratio=candidate.bottleneck_ratio,
        )
    if candidate.architecture == "grouped_residual":
        return WideResidualNoBNSR(
            channels=candidate.channels,
            num_blocks=candidate.num_blocks,
            slope=candidate.slope,
            body_kernel_size=candidate.body_kernel_size,
            groups=candidate.groups,
        )
    if candidate.architecture == "two_stage_residual":
        return TwoStageResidualNoBNSR(
            channels=candidate.channels,
            num_blocks=candidate.num_blocks,
            slope=candidate.slope,
            stages=max(1, candidate.stages),
        )
    raise KeyError(f"Unsupported architecture={candidate.architecture}")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def operator_audit(model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for module in model.modules():
        if len(list(module.children())) != 0:
            continue
        name = module.__class__.__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def validate_operator_audit(model: nn.Module) -> None:
    unexpected = []
    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            module_name = module.__class__.__name__
            if module_name not in SAFE_LEAF_MODULES:
                unexpected.append(f"{name}: {module_name}")
    if unexpected:
        raise RuntimeError(f"Unexpected leaf modules found: {unexpected}")


def verify_model_contract(model: nn.Module, eval_size: int) -> dict[str, Any]:
    validate_operator_audit(model)
    model_device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, eval_size, eval_size, dtype=torch.float32, device=model_device)
    with torch.no_grad():
        output = model(dummy)
    if tuple(output.shape) != (1, 3, eval_size, eval_size):
        raise RuntimeError(f"Model contract failed: expected (1, 3, {eval_size}, {eval_size}), got {tuple(output.shape)}")
    return {
        "input_shape": list(dummy.shape),
        "output_shape": list(output.shape),
        "params": count_parameters(model),
        "module_ops": operator_audit(model),
        "safe_leaf_modules": list(SAFE_LEAF_MODULES),
    }


def move_batch(
    lr_img: torch.Tensor,
    hr_img: torch.Tensor,
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    lr_img = lr_img.to(device, non_blocking=True)
    hr_img = hr_img.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        lr_img = lr_img.contiguous(memory_format=torch.channels_last)
        hr_img = hr_img.contiguous(memory_format=torch.channels_last)
    return lr_img, hr_img


def residual_target_l1_loss(sr_pred: torch.Tensor, lr_img: torch.Tensor, hr_img: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(sr_pred - lr_img, hr_img - lr_img)


def verify_residual_l1_batch(model: nn.Module, loader: DataLoader, device: torch.device, cfg: PipelineConfig) -> None:
    lr_img, hr_img, _ = next(iter(loader))
    lr_img, hr_img = move_batch(lr_img, hr_img, device, cfg.channels_last)
    with torch.no_grad():
        pred = model(lr_img)
    torch.testing.assert_close(residual_target_l1_loss(pred, lr_img, hr_img), F.l1_loss(pred - lr_img, hr_img - lr_img))


def create_ema_model(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    return ema_model


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for key, value in ema_state.items():
        value.copy_(value * decay + model_state[key] * (1.0 - decay))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    best_metric: float,
    cfg: PipelineConfig,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": best_metric,
            "config": cfg.as_json(),
            "saved_at": time.time(),
        },
        path,
    )


def load_checkpoint(model: nn.Module, checkpoint_path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint


def make_grad_scaler(device: torch.device, use_amp: bool):
    if not (use_amp and device.type == "cuda"):
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def lr_multiplier(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, cfg: PipelineConfig) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_input_psnr = 0.0
    total_delta = 0.0
    sample_count = 0
    for lr_img, hr_img, _ in loader:
        lr_img, hr_img = move_batch(lr_img, hr_img, device, cfg.channels_last)
        with autocast_context(device, cfg.use_amp):
            pred = model(lr_img)
            loss = residual_target_l1_loss(pred, lr_img, hr_img)
        pred_psnr = tensor_psnr(pred, hr_img)
        input_psnr = tensor_psnr(lr_img, hr_img)
        delta = pred_psnr - input_psnr
        batch_size = lr_img.size(0)
        total_loss += float(loss.item()) * batch_size
        total_psnr += pred_psnr.sum().item()
        total_input_psnr += input_psnr.sum().item()
        total_delta += delta.sum().item()
        sample_count += batch_size
    return {
        "val_loss": total_loss / max(1, sample_count),
        "val_psnr": total_psnr / max(1, sample_count),
        "input_psnr": total_input_psnr / max(1, sample_count),
        "delta_psnr": total_delta / max(1, sample_count),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: PipelineConfig,
    layout: RunLayout,
) -> tuple[Path, dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_multiplier(epoch, cfg.num_epochs, cfg.warmup_epochs),
    )
    scaler = make_grad_scaler(device, cfg.use_amp)
    ema_model = create_ema_model(model)
    best_checkpoint = layout.checkpoint_dir / "best.pt"
    last_checkpoint = layout.checkpoint_dir / "last.pt"
    best_val_psnr = float("-inf")
    best_epoch = 0
    latest_metrics: dict[str, Any] = {}

    for epoch in range(cfg.num_epochs):
        epoch_index = epoch + 1
        start_time = time.time()
        train_metrics = train_one_epoch(model, ema_model, train_loader, optimizer, scaler, device, cfg)
        eval_metrics = evaluate_loader(ema_model, val_loader, device, cfg)
        scheduler.step()
        latest_metrics = {
            "epoch": epoch_index,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.time() - start_time,
            **train_metrics,
            **eval_metrics,
        }
        append_jsonl(layout.metrics_path, latest_metrics)
        save_json(
            layout.status_path,
            {
                "phase": "training",
                "epoch": epoch_index,
                "best_epoch": best_epoch,
                "best_val_psnr": best_val_psnr,
                "latest_metrics": latest_metrics,
            },
        )
        save_checkpoint(last_checkpoint, ema_model, optimizer, epoch_index, latest_metrics, best_val_psnr, cfg)
        if eval_metrics["val_psnr"] > best_val_psnr:
            best_val_psnr = float(eval_metrics["val_psnr"])
            best_epoch = epoch_index
            save_checkpoint(best_checkpoint, ema_model, optimizer, epoch_index, latest_metrics, best_val_psnr, cfg)
        print(
            f"Epoch {epoch_index:03d}/{cfg.num_epochs:03d} | "
            f"lr {latest_metrics['learning_rate']:.2e} | "
            f"train_loss {train_metrics['train_loss']:.6f} | "
            f"train_psnr {train_metrics['train_psnr']:.4f} | "
            f"val_psnr {eval_metrics['val_psnr']:.4f} | "
            f"baseline {eval_metrics['input_psnr']:.4f} | "
            f"delta {eval_metrics['delta_psnr']:+.4f} | "
            f"time {latest_metrics['epoch_seconds']:.1f}s"
        )

    final_summary = {
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "best_epoch": best_epoch,
        "best_val_psnr": best_val_psnr,
        "final_metrics": latest_metrics,
    }
    save_json(layout.summary_path, {"phase": "training_complete", "config": cfg.as_json(), "training": final_summary})
    return best_checkpoint, final_summary


def train_one_epoch(
    model: nn.Module,
    ema_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    cfg: PipelineConfig,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    sample_count = 0
    for lr_img, hr_img, _ in loader:
        lr_img, hr_img = move_batch(lr_img, hr_img, device, cfg.channels_last)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast_context(device, cfg.use_amp):
                pred = model(lr_img)
                loss = residual_target_l1_loss(pred, lr_img, hr_img)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with autocast_context(device, cfg.use_amp):
                pred = model(lr_img)
                loss = residual_target_l1_loss(pred, lr_img, hr_img)
            loss.backward()
            optimizer.step()
        ema_update(ema_model, model, cfg.ema_decay)
        batch_size = lr_img.size(0)
        total_loss += float(loss.item()) * batch_size
        with torch.no_grad():
            total_psnr += tensor_psnr(pred.detach(), hr_img).sum().item()
        sample_count += batch_size
    return {
        "train_loss": total_loss / max(1, sample_count),
        "train_psnr": total_psnr / max(1, sample_count),
    }


def export_to_onnx(model: nn.Module, onnx_path: Path, device: torch.device, eval_size: int, verify: bool) -> dict[str, Any]:
    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, eval_size, eval_size, device=device)
    with torch.no_grad():
        export_kwargs = {
            "export_params": True,
            "opset_version": 13,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": None,
        }
        try:
            torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)
    metadata: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "onnx_size_kb": round(onnx_path.stat().st_size / 1024.0, 2),
    }
    if onnx is not None:
        loaded = onnx.load(str(onnx_path))
        onnx.checker.check_model(loaded)
        metadata["onnx_checker"] = "passed"
        metadata["onnx_opset"] = int(loaded.opset_import[0].version)
    if verify and ort is not None:
        cpu_dummy = dummy.detach().cpu().numpy()
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_output = session.run(None, {"input": cpu_dummy})[0]
        torch_output = model(dummy).detach().cpu().numpy()
        diff = np.abs(ort_output - torch_output)
        metadata["ort_max_diff"] = float(diff.max())
        metadata["ort_mean_diff"] = float(diff.mean())
    return metadata


def compute_image_profile(image_path: Path) -> dict[str, float]:
    image = pil_rgb(image_path).resize((64, 64), RESAMPLING_BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    return {
        "brightness": float(gray.mean()),
        "texture": float((grad_x + grad_y) * 0.5),
    }


def select_calibration_pairs(train_pairs: list[tuple[Path, Path, str]], count: int) -> list[tuple[Path, Path, str]]:
    if not train_pairs:
        return []
    if len(train_pairs) <= count:
        return train_pairs
    profiled: list[tuple[float, tuple[Path, Path, str]]] = []
    for lr_path, hr_path, name in train_pairs:
        stats = compute_image_profile(lr_path)
        score = stats["brightness"] * 0.4 + stats["texture"] * 0.6
        profiled.append((score, (lr_path, hr_path, name)))
    profiled.sort(key=lambda item: item[0])
    if count == 1:
        return [profiled[len(profiled) // 2][1]]
    selected: list[tuple[Path, Path, str]] = []
    for index in range(count):
        position = round(index * (len(profiled) - 1) / (count - 1))
        selected.append(profiled[position][1])
    return selected


def export_calibration_dataset(
    train_pairs: list[tuple[Path, Path, str]],
    calibration_dir: Path,
    eval_size: int,
    calibration_count: int,
) -> dict[str, Any]:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    selected = select_calibration_pairs(train_pairs, calibration_count)
    manifest: list[dict[str, Any]] = []
    for index, (lr_path, _, name) in enumerate(selected):
        image = pil_rgb(lr_path)
        if image.size != (eval_size, eval_size):
            image = ImageOps.fit(image, (eval_size, eval_size), method=RESAMPLING_BICUBIC)
        file_name = f"{index:03d}_{Path(name).stem}.png"
        output_path = calibration_dir / file_name
        image.save(output_path)
        manifest.append(
            {
                "index": index,
                "name": name,
                "source_lr": str(lr_path),
                "image_path": str(output_path),
                "derived_from_training": True,
            }
        )
    manifest_path = calibration_dir / "manifest.json"
    save_json(
        manifest_path,
        {
            "count": len(manifest),
            "source": "training_pairs",
            "eval_size": eval_size,
            "items": manifest,
        },
    )
    return {
        "calibration_dir": str(calibration_dir),
        "manifest_path": str(manifest_path),
        "count": len(manifest),
        "source": "training_pairs",
        "derived_from_training": True,
    }


def build_mxq_handoff(
    cfg: PipelineConfig,
    onnx_path: Path,
    calibration_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    helper_path = cfg.project_root / "lab3_step2_onnx_to_mxq.py"
    if not helper_path.exists():
        return {
            "status": "missing_helper",
            "helper_path": str(helper_path),
            "onnx_path": str(onnx_path),
            "calibration_dir": str(calibration_dir),
            "output": str(output_path),
        }
    command = [
        sys.executable,
        str(helper_path),
        "--onnx",
        str(onnx_path),
        "--calibration-dir",
        str(calibration_dir),
        "--output",
        str(output_path),
    ]
    if cfg.compile_command_template:
        command.extend(["--command-template", cfg.compile_command_template])
    for arg in cfg.compile_extra_args:
        command.extend(["--extra-arg", arg])
    if not cfg.run_mxq_compile:
        command.append("--dry-run")
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(cfg.project_root))
    payload: dict[str, Any] = {
        "helper_path": str(helper_path),
        "status": "failed_to_parse",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass
    if "helper_path" not in payload:
        payload["helper_path"] = str(helper_path)
    payload["returncode"] = completed.returncode
    payload["stderr"] = completed.stderr.strip()
    payload["requested_compile"] = cfg.run_mxq_compile
    payload["output_exists"] = output_path.exists()
    return payload


def summarize_gates(summary: dict[str, Any], prior_best_val_psnr: float | None, tolerance: float) -> dict[str, Any]:
    evaluation = summary.get("evaluation", {})
    contract = summary.get("model_contract", {})
    calibration = summary.get("calibration", {})
    onnx_summary = summary.get("onnx", {})
    mxq_summary = summary.get("mxq", {})

    val_psnr = evaluation.get("val_psnr")
    material_regression = False
    if prior_best_val_psnr is not None and val_psnr is not None:
        material_regression = val_psnr < (prior_best_val_psnr - tolerance)

    screening_pass = all(key in evaluation for key in ["val_psnr", "input_psnr", "delta_psnr"])
    contract_pass = contract.get("input_shape") == [1, 3, 256, 256] and contract.get("output_shape") == [1, 3, 256, 256]
    safe_ops_pass = set(contract.get("module_ops", {}).keys()).issubset(set(SAFE_LEAF_MODULES))
    onnx_pass = bool(onnx_summary.get("onnx_path")) and onnx_summary.get("onnx_checker") == "passed"
    if "ort_max_diff" in onnx_summary:
        onnx_pass = onnx_pass and onnx_summary["ort_max_diff"] < 1e-3
    calibration_pass = calibration.get("derived_from_training") is True and calibration.get("count", 0) > 0
    mxq_handoff_pass = mxq_summary.get("status") in {"handoff_only", "dry_run", "completed"}
    promotion_pass = all([screening_pass, contract_pass, safe_ops_pass, onnx_pass, calibration_pass, mxq_handoff_pass]) and not material_regression

    return {
        "screening_pass": screening_pass,
        "contract_pass": contract_pass,
        "safe_ops_pass": safe_ops_pass,
        "onnx_pass": onnx_pass,
        "calibration_pass": calibration_pass,
        "mxq_handoff_pass": mxq_handoff_pass,
        "material_regression": material_regression,
        "promotion_pass": promotion_pass,
        "prior_best_val_psnr": prior_best_val_psnr,
        "tolerance": tolerance,
    }


def print_artifact_summary(summary: dict[str, Any]) -> None:
    print("Artifact summary:")
    for key in ["best_checkpoint", "onnx_path", "summary_path", "report_path", "notebook_path"]:
        if key in summary:
            print(f"  {key}: {summary[key]}")


def run_pipeline(cfg: PipelineConfig, prior_best_val_psnr: float | None = None) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = resolve_device()
    layout = run_layout_from_config(cfg)
    save_json(layout.config_path, cfg.as_json())

    train_pairs = collect_train_pairs(cfg.data_root, cfg.train_pair_limit)
    val_pairs = collect_val_pairs(cfg.data_root, cfg.val_pair_limit)
    if not train_pairs:
        raise FileNotFoundError(f"No paired training pairs found under {cfg.data_root}")
    if not val_pairs:
        raise FileNotFoundError(f"No paired validation pairs found under {cfg.data_root}")

    train_loader, val_loader = make_dataloaders(train_pairs, val_pairs, cfg, device)
    model = build_model(cfg).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    contract = verify_model_contract(model, cfg.eval_size)
    verify_residual_l1_batch(model, train_loader, device, cfg)

    best_checkpoint = layout.checkpoint_dir / "best.pt"
    training_summary: dict[str, Any] = {}
    if cfg.run_training:
        best_checkpoint, training_summary = train_model(model, train_loader, val_loader, device, cfg, layout)
    elif not best_checkpoint.exists():
        raise FileNotFoundError(f"RUN_TRAINING=False but checkpoint is missing: {best_checkpoint}")
    else:
        training_summary = {"best_checkpoint": str(best_checkpoint), "resumed_without_training": True}

    eval_model = build_model(cfg).to(device)
    load_checkpoint(eval_model, best_checkpoint, map_location=device)
    if cfg.channels_last and device.type == "cuda":
        eval_model = eval_model.to(memory_format=torch.channels_last)
    evaluation_summary = evaluate_loader(eval_model, val_loader, device, cfg)

    onnx_path = layout.export_dir / "best.onnx"
    onnx_summary: dict[str, Any] = {}
    if cfg.run_onnx_export:
        onnx_summary = export_to_onnx(eval_model, onnx_path, device, cfg.eval_size, cfg.verify_onnx_export)

    calibration_summary = export_calibration_dataset(
        train_pairs=train_pairs,
        calibration_dir=layout.calibration_dir,
        eval_size=cfg.eval_size,
        calibration_count=cfg.calibration_count,
    )
    mxq_summary = build_mxq_handoff(cfg, onnx_path, layout.calibration_dir, layout.mxq_path)

    summary = {
        "device": str(device),
        "backend": cfg.backend,
        "config": cfg.as_json(),
        "run_day": layout.run_day,
        "run_root": str(layout.run_root),
        "artifact_root": str(layout.artifact_root),
        "runs_root": str(layout.runs_root),
        "day_root": str(layout.day_root),
        "execution": {
            "backend": cfg.backend,
            "started_day": layout.run_day,
            "modal_app_name": cfg.modal_app_name,
            "modal_function_name": cfg.modal_function_name,
            "modal_gpu": cfg.modal_gpu,
            "modal_timeout_minutes": cfg.modal_timeout_minutes,
            "modal_data_volume": cfg.modal_data_volume,
            "modal_runs_volume": cfg.modal_runs_volume,
            "synced_local_run_root": cfg.synced_local_run_root,
        },
        "pair_summary": summarize_pairs(train_pairs, val_pairs),
        "candidate": cfg.candidate().as_json(),
        "model_contract": contract,
        "training": training_summary,
        "evaluation": evaluation_summary,
        "onnx": onnx_summary,
        "calibration": calibration_summary,
        "mxq": mxq_summary,
        "best_checkpoint": str(best_checkpoint),
        "onnx_path": str(onnx_path),
        "mxq_path": str(layout.mxq_path),
        "summary_path": str(layout.summary_path),
        "report_path": str(layout.report_path),
        "notebook_path": str(layout.notebook_path),
    }
    summary["gates"] = summarize_gates(summary, prior_best_val_psnr, cfg.material_regression_tolerance)
    save_json(layout.summary_path, {"phase": "pipeline_complete", "config": cfg.as_json(), "summary": summary})
    save_json(
        layout.status_path,
        {
            "phase": "pipeline_complete",
            "candidate_id": cfg.candidate_id,
            "summary_path": str(layout.summary_path),
            "promotion_pass": summary["gates"]["promotion_pass"],
        },
    )
    save_json(layout.report_path, summary)
    return summary


def default_data_root(project_root: Path) -> Path:
    return normalize_path(os.environ.get("LAB3_DATA_ROOT", project_root / "Data"))


def default_run_name(candidate_id: str = "wide_residual_nobn_v1") -> str:
    return f"lab3_{candidate_id}_{time.strftime('%Y%m%d_%H%M%S')}"


def comparison_signature_from_cfg(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "train_pairs": cfg.train_pair_limit,
        "val_pairs": cfg.val_pair_limit,
        "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "eval_size": cfg.eval_size,
        "train_patch_size": cfg.train_patch_size,
        "backend": cfg.backend,
        "modal_gpu": cfg.modal_gpu or None,
    }


def comparison_signature_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    execution = summary.get("execution", {})
    pair_summary = summary.get("pair_summary", {})
    config = summary.get("config", {})
    training = summary.get("training", {})
    final_metrics = training.get("final_metrics", {})
    return {
        "train_pairs": pair_summary.get("train_pairs", config.get("train_pair_limit")),
        "val_pairs": pair_summary.get("val_pairs", config.get("val_pair_limit")),
        "num_epochs": config.get("num_epochs", final_metrics.get("epoch")),
        "batch_size": config.get("batch_size"),
        "eval_size": config.get("eval_size"),
        "train_patch_size": config.get("train_patch_size"),
        "backend": summary.get("backend", execution.get("backend")),
        "modal_gpu": execution.get("modal_gpu") or config.get("modal_gpu"),
    }


def summary_matches_signature(summary: dict[str, Any], signature: dict[str, Any]) -> bool:
    current = comparison_signature_from_summary(summary)
    for key, value in signature.items():
        if value is None:
            continue
        if current.get(key) != value:
            return False
    return True


def best_val_psnr_for_signature(history: list[dict[str, Any]], signature: dict[str, Any]) -> float | None:
    matches = [
        summary.get("evaluation", {}).get("val_psnr")
        for summary in history
        if summary_matches_signature(summary, signature)
        and summary.get("evaluation", {}).get("val_psnr") is not None
    ]
    return max(matches) if matches else None


def iter_summary_paths(artifact_root: Path) -> list[Path]:
    runs_root = artifact_root / "runs"
    if not runs_root.exists():
        return []
    nested = sorted(runs_root.glob("*/*/summary.json"))
    legacy = sorted(runs_root.glob("*/summary.json"))
    return nested + [path for path in legacy if path not in nested]


def load_run_summaries(artifact_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for summary_path in iter_summary_paths(normalize_path(artifact_root)):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = payload.get("summary")
        if isinstance(summary, dict):
            if "config" not in summary and isinstance(payload.get("config"), dict):
                summary["config"] = payload["config"]
            summaries.append(summary)
    return summaries
