from __future__ import annotations

import random
import sys
from multiprocessing import Value
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
RESAMPLING_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
VAL_SIZE = 256
# Mix epoch into RNG seed so each (index, epoch) gets different crop/aug; large prime avoids collisions.
_EPOCH_RNG_STRIDE = 1_000_000_007


def _ensure_lab3_tools_on_path(project_root: Path) -> None:
    tools = project_root / "tools"
    if tools.is_dir() and str(tools) not in sys.path:
        sys.path.insert(0, str(tools))


def validate_data_layout(data_root: Path, project_root: Path) -> None:
    """Fail fast if expected Lab3 folders are missing."""
    required_train_hr = [data_root / "HR_train" / f"HR_train{i}" for i in range(1, 5)]
    required_train_lr = [data_root / "LR_train" / f"LR_train{i}" for i in range(1, 5)]
    missing = [p for p in required_train_hr + required_train_lr if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing required training subfolders:\n  " + "\n  ".join(str(p) for p in missing)
        )
    hr_val = data_root / "HR_val"
    lr_val = data_root / "LR_val"
    if not hr_val.is_dir():
        raise FileNotFoundError(f"Missing validation HR folder: {hr_val}")
    if not lr_val.is_dir():
        raise FileNotFoundError(f"Missing validation LR folder: {lr_val}")
    _ = project_root  # reserved for future checks


def collect_train_pairs(data_root: Path, project_root: Path) -> list[tuple[Path, Path, str]]:
    _ensure_lab3_tools_on_path(project_root)
    from lab3_pipeline_lib import collect_train_pairs as _ctp

    return _ctp(data_root, limit=None)


def collect_val_pairs(data_root: Path, project_root: Path) -> list[tuple[Path, Path, str]]:
    _ensure_lab3_tools_on_path(project_root)
    from lab3_pipeline_lib import collect_val_pairs as _cvp

    return _cvp(data_root, limit=None)


def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor_chw01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def random_crop_pair(
    lr_img: Image.Image, hr_img: Image.Image, size: int, rng: random.Random
) -> tuple[Image.Image, Image.Image]:
    width, height = lr_img.size
    if min(width, height) < size:
        lr_img = ImageOps.fit(lr_img, (size, size), method=RESAMPLING_BICUBIC)
        hr_img = ImageOps.fit(hr_img, (size, size), method=RESAMPLING_BICUBIC)
        return lr_img, hr_img
    left = rng.randint(0, width - size)
    top = rng.randint(0, height - size)
    box = (left, top, left + size, top + size)
    return lr_img.crop(box), hr_img.crop(box)


def augment_pair(
    lr_img: Image.Image, hr_img: Image.Image, rng: random.Random
) -> tuple[Image.Image, Image.Image]:
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


class PairedImageDataset(Dataset):
    """Paired LR/HR by basename; train uses synchronized crop + aug; val uses full 256x256 only."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path, str]],
        *,
        train: bool,
        seed: int,
        patch_size: int,
        train_epoch_counter: Value | None = None,
    ):
        self.pairs = pairs
        self.train = train
        self.seed = seed
        self.patch_size = patch_size
        # Shared int updated each training epoch so augmentations vary across epochs with num_workers>0.
        self._train_epoch_counter = train_epoch_counter if train else None

    def __len__(self) -> int:
        return len(self.pairs)

    def _rng_seed(self, index: int) -> int:
        if not self.train or self._train_epoch_counter is None:
            return self.seed + index
        ep = int(self._train_epoch_counter.value)
        return self.seed + index + ep * _EPOCH_RNG_STRIDE

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        lr_path, hr_path, name = self.pairs[index]
        lr_img = pil_rgb(lr_path)
        hr_img = pil_rgb(hr_path)
        if self.train:
            rng = random.Random(self._rng_seed(index))
            lr_img, hr_img = random_crop_pair(lr_img, hr_img, self.patch_size, rng)
            lr_img, hr_img = augment_pair(lr_img, hr_img, rng)
        else:
            if lr_img.size != (VAL_SIZE, VAL_SIZE) or hr_img.size != (VAL_SIZE, VAL_SIZE):
                raise ValueError(
                    f"Validation pair {name} must be {VAL_SIZE}x{VAL_SIZE}; "
                    f"got LR {lr_img.size} HR {hr_img.size}"
                )
        return pil_to_tensor_chw01(lr_img), pil_to_tensor_chw01(hr_img), name


def make_dataloaders(
    train_pairs: list[tuple[Path, Path, str]],
    val_pairs: list[tuple[Path, Path, str]],
    *,
    batch_size: int,
    patch_size: int,
    seed: int,
    train_workers: int,
    val_workers: int,
    device: torch.device,
    train_epoch_counter: Value | None = None,
) -> tuple[DataLoader, DataLoader]:
    pin = device.type == "cuda"
    train_loader = DataLoader(
        PairedImageDataset(
            train_pairs,
            train=True,
            seed=seed,
            patch_size=patch_size,
            train_epoch_counter=train_epoch_counter,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin,
        drop_last=False,
    )
    val_bs = max(1, min(batch_size, 8))
    val_loader = DataLoader(
        PairedImageDataset(val_pairs, train=False, seed=seed, patch_size=patch_size),
        batch_size=val_bs,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin,
        drop_last=False,
    )
    return train_loader, val_loader


def summarize_pairs(
    train_pairs: list[tuple[Path, Path, str]], val_pairs: list[tuple[Path, Path, str]]
) -> dict[str, Any]:
    return {
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_preview": [n for _, _, n in train_pairs[:3]],
        "val_preview": [n for _, _, n in val_pairs[:3]],
    }
