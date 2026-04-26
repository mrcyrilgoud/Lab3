#!/usr/bin/env python3
"""CLI entrypoint for U-Net Experiment 1 on Modal.

Generated from `u_net_experiment_1_modal_app.ipynb` cells 2, 4, 6, and 8.

Use `modal run` so Modal loads this module normally (avoids Jupyter cloudpickle issues,
notably Python 3.13 + PyTorch `GenericModule` serialization failures).

Example (smoke):
  cd "U-Net Experiment 1"
  UNET_EXPERIMENT_EXECUTE_MODAL=true UNET_EXPERIMENT_RUN_MODE=smoke \
    UNET_EXPERIMENT_MODAL_GPU=A10G python3 -m modal run u_net_experiment_1_modal_run.py
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import modal
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

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}
RESAMPLING_BICUBIC = getattr(Image, 'Resampling', Image).BICUBIC
EXPECTED_TRAIN_PAIRS = 3036
EXPECTED_VAL_PAIRS = 100
KNOWN_LEAF_MODULES = {'Conv2d', 'ConvTranspose2d', 'LeakyReLU', 'PReLU', 'ReLU', 'Identity'}
KNOWN_SUPPORTED_OR_FALLBACK_ONNX_OPS = {
    'Conv', 'ConvTranspose', 'LeakyRelu', 'PRelu', 'Relu', 'HardSigmoid', 'HardSwish',
    'Add', 'Concat', 'Cast', 'Clip', 'Div', 'Flatten', 'Gather', 'GatherND', 'MatMul',
    'Mul', 'ReduceMean', 'Resize', 'Reshape', 'Sigmoid', 'Slice', 'Softmax', 'Split',
    'Squeeze', 'Sub', 'Tanh', 'Tile', 'Transpose', 'Unsqueeze', 'Where', 'Pad',
    'MaxPool', 'AveragePool', 'GlobalAveragePool', 'Identity', 'Constant'
}

NOTEBOOK_DIR = Path.cwd().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'U-Net Experiment 1' else NOTEBOOK_DIR
DATA_ROOT = PROJECT_ROOT / 'Data'
LOCAL_RUNS_ROOT = PROJECT_ROOT / 'runs'
REMOTE_DATA_MOUNT = PurePosixPath('/mnt/lab3-data')
REMOTE_RUNS_MOUNT = PurePosixPath('/mnt/lab3-runs')
REMOTE_DATA_ROOT = REMOTE_DATA_MOUNT / 'Data'
REMOTE_RUNS_ROOT = REMOTE_RUNS_MOUNT / 'runs'
REMOTE_PYTHON_VERSION = os.environ.get("UNET_EXPERIMENT_MODAL_PYTHON", "3.11")
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def tensor_psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((a.float().clamp(0.0, 1.0) - b.float().clamp(0.0, 1.0)) ** 2, dim=(-3, -2, -1))
    mse = mse.clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def summarize_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    total_hr = sum(int(item['hr_count']) for item in items)
    total_lr = sum(int(item['lr_count']) for item in items)
    total_paired = sum(int(item['paired_count']) for item in items)
    total_hr_only = sum(int(item['hr_only_count']) for item in items)
    total_lr_only = sum(int(item['lr_only_count']) for item in items)
    return {
        'hr_total': total_hr,
        'lr_total': total_lr,
        'paired_total': total_paired,
        'hr_only_total': total_hr_only,
        'lr_only_total': total_lr_only,
    }


def _file_maps(directory: Path) -> tuple[dict[str, Path], list[str]]:
    if not directory.exists():
        return {}, []
    file_map: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in file_map:
            duplicates.append(path.stem)
        else:
            file_map[path.stem] = path
    return file_map, duplicates


def audit_pair_directory(hr_dir: Path, lr_dir: Path, split_name: str) -> dict[str, Any]:
    if not hr_dir.exists():
        raise FileNotFoundError(f'Missing HR directory for {split_name}: {hr_dir}')
    if not lr_dir.exists():
        raise FileNotFoundError(f'Missing LR directory for {split_name}: {lr_dir}')

    hr_map, hr_duplicates = _file_maps(hr_dir)
    lr_map, lr_duplicates = _file_maps(lr_dir)
    shared = sorted(set(hr_map) & set(lr_map))
    hr_only = sorted(set(hr_map) - set(lr_map))
    lr_only = sorted(set(lr_map) - set(hr_map))

    size_mismatches: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for stem in shared:
        try:
            with Image.open(hr_map[stem]) as hr_img_raw, Image.open(lr_map[stem]) as lr_img_raw:
                hr_img = hr_img_raw.convert('RGB')
                lr_img = lr_img_raw.convert('RGB')
                if hr_img.size != lr_img.size:
                    size_mismatches.append(
                        {
                            'stem': stem,
                            'hr_size': list(hr_img.size),
                            'lr_size': list(lr_img.size),
                        }
                    )
        except Exception as exc:
            unreadable.append({'stem': stem, 'error': str(exc)})

    result = {
        'split': split_name,
        'hr_dir': str(hr_dir),
        'lr_dir': str(lr_dir),
        'hr_count': len(hr_map),
        'lr_count': len(lr_map),
        'paired_count': len(shared),
        'hr_only_count': len(hr_only),
        'lr_only_count': len(lr_only),
        'duplicate_hr_basenames': hr_duplicates,
        'duplicate_lr_basenames': lr_duplicates,
        'hr_only_examples': hr_only[:10],
        'lr_only_examples': lr_only[:10],
        'checked_size_pairs': len(shared),
        'size_mismatches': size_mismatches[:10],
        'unreadable_pairs': unreadable[:10],
        'passed': not any([hr_duplicates, lr_duplicates, hr_only, lr_only, size_mismatches, unreadable]),
    }
    return result


def run_pairing_audit(data_root: Path) -> dict[str, Any]:
    train_results = []
    hr_train_root = data_root / 'HR_train'
    lr_train_root = data_root / 'LR_train'
    expected_train_dirs = [f'HR_train{i}' for i in range(1, 5)]
    for hr_name in expected_train_dirs:
        suffix = hr_name.replace('HR_train', '')
        lr_name = f'LR_train{suffix}'
        train_results.append(
            audit_pair_directory(
                hr_train_root / hr_name,
                lr_train_root / lr_name,
                split_name=hr_name,
            )
        )

    val_result = audit_pair_directory(
        data_root / 'HR_val',
        data_root / 'LR_val',
        split_name='val',
    )

    train_totals = summarize_counts(train_results)
    expected = {
        'train_pairs_expected': EXPECTED_TRAIN_PAIRS,
        'val_pairs_expected': EXPECTED_VAL_PAIRS,
        'known_good_baseline': {
            'train_pairing': 'HR_train1-4 and LR_train1-4 pair cleanly',
            'val_pairing': 'HR_val and LR_val pair cleanly',
            'size_check': 'full paired image size and RGB readability checks passed during the last known-good audit',
        },
    }
    observed = {
        'train_pairs_observed': train_totals['paired_total'],
        'val_pairs_observed': val_result['paired_count'],
    }
    passed = all(item['passed'] for item in train_results) and val_result['passed']
    report = {
        'expected': expected,
        'observed': observed,
        'train_splits': train_results,
        'val_split': val_result,
        'passed': passed,
    }
    if not passed:
        raise RuntimeError('HR/LR pairing audit failed. Inspect the printed report before proceeding.')
    return report


def collect_train_pairs(data_root: Path, limit: int | None = None) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    for index in range(1, 5):
        hr_dir = data_root / 'HR_train' / f'HR_train{index}'
        lr_dir = data_root / 'LR_train' / f'LR_train{index}'
        hr_map, _ = _file_maps(hr_dir)
        lr_map, _ = _file_maps(lr_dir)
        for stem in sorted(set(hr_map) & set(lr_map)):
            pairs.append((lr_map[stem], hr_map[stem], f'HR_train{index}/{stem}'))
    return pairs if limit is None else pairs[:limit]


def collect_val_pairs(data_root: Path, limit: int | None = None) -> list[tuple[Path, Path, str]]:
    hr_map, _ = _file_maps(data_root / 'HR_val')
    lr_map, _ = _file_maps(data_root / 'LR_val')
    pairs = [(lr_map[stem], hr_map[stem], stem) for stem in sorted(set(hr_map) & set(lr_map))]
    return pairs if limit is None else pairs[:limit]


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
    turns = rng.randint(0, 3)
    if turns:
        angle = 90 * turns
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


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def move_batch(lr_img: torch.Tensor, hr_img: torch.Tensor, device: torch.device, channels_last: bool) -> tuple[torch.Tensor, torch.Tensor]:
    lr_img = lr_img.to(device, non_blocking=True)
    hr_img = hr_img.to(device, non_blocking=True)
    if channels_last and device.type == 'cuda':
        lr_img = lr_img.contiguous(memory_format=torch.channels_last)
        hr_img = hr_img.contiguous(memory_format=torch.channels_last)
    return lr_img, hr_img


def residual_target_l1_loss(sr_pred: torch.Tensor, lr_img: torch.Tensor, hr_img: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(sr_pred - lr_img, hr_img - lr_img)


def make_activation(slope: float) -> nn.Module:
    return nn.LeakyReLU(negative_slope=slope, inplace=True)


def init_tail_small(layer: nn.Conv2d, scale: float = 1e-3) -> None:
    nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
    layer.weight.data.mul_(scale)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, slope: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            make_activation(slope),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            make_activation(slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleBridgeUNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 56,
        levels: int = 4,
        slope: float = 0.10,
        use_deep_skip_bridges: bool = False,
    ):
        super().__init__()
        if levels != 4:
            raise ValueError('This notebook expects levels=4 for the default contract.')
        self.use_deep_skip_bridges = use_deep_skip_bridges
        ch = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, ch[0], 3, padding=1, bias=True), make_activation(slope))
        self.enc1 = DoubleConv(ch[0], ch[0], slope)
        self.down1 = nn.Sequential(nn.Conv2d(ch[0], ch[1], 3, stride=2, padding=1, bias=True), make_activation(slope))
        self.enc2 = DoubleConv(ch[1], ch[1], slope)
        self.down2 = nn.Sequential(nn.Conv2d(ch[1], ch[2], 3, stride=2, padding=1, bias=True), make_activation(slope))
        self.enc3 = DoubleConv(ch[2], ch[2], slope)
        self.down3 = nn.Sequential(nn.Conv2d(ch[2], ch[3], 3, stride=2, padding=1, bias=True), make_activation(slope))
        self.bottleneck = DoubleConv(ch[3], ch[3], slope)
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2, bias=True)
        self.dec3 = DoubleConv(ch[2], ch[2], slope)
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2, bias=True)
        self.dec2 = DoubleConv(ch[1], ch[1], slope)
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2, bias=True)
        self.dec1 = DoubleConv(ch[0], ch[0], slope)
        self.tail = nn.Conv2d(ch[0], 3, 3, padding=1, bias=True)
        init_tail_small(self.tail)

    def predict_delta(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3_input = self.up3(b)
        if self.use_deep_skip_bridges:
            d3_input = d3_input + e3
        d3 = self.dec3(d3_input)
        d2_input = self.up2(d3)
        if self.use_deep_skip_bridges:
            d2_input = d2_input + e2
        d2 = self.dec2(d2_input)
        d1 = self.dec1(self.up1(d2) + e1)
        return self.tail(d1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.predict_delta(x)


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


def validate_leaf_modules(model: nn.Module) -> dict[str, Any]:
    unexpected = []
    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            module_name = module.__class__.__name__
            if module_name not in KNOWN_LEAF_MODULES:
                unexpected.append(f'{name}: {module_name}')
    return {
        'known_leaf_modules': sorted(KNOWN_LEAF_MODULES),
        'unexpected_leaf_modules': unexpected,
        'all_leaf_modules_known': not unexpected,
    }


def verify_model_contract(model: nn.Module, eval_size: int) -> dict[str, Any]:
    leaf_audit = validate_leaf_modules(model)
    params = count_parameters(model)
    model_device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, eval_size, eval_size, dtype=torch.float32, device=model_device)
    with torch.no_grad():
        out = model(dummy)
    if tuple(out.shape) != (1, 3, eval_size, eval_size):
        raise RuntimeError(f'Contract failure: expected (1, 3, {eval_size}, {eval_size}), got {tuple(out.shape)}')
    return {
        'input_shape': list(dummy.shape),
        'output_shape': list(out.shape),
        'params': params,
        'module_ops': operator_audit(model),
        'leaf_audit': leaf_audit,
    }


def verify_residual_l1_batch(model: nn.Module, loader: DataLoader, device: torch.device, channels_last: bool) -> None:
    lr_img, hr_img, _ = next(iter(loader))
    lr_img, hr_img = move_batch(lr_img, hr_img, device, channels_last)
    with torch.no_grad():
        pred = model(lr_img)
    torch.testing.assert_close(residual_target_l1_loss(pred, lr_img, hr_img), F.l1_loss(pred - lr_img, hr_img - lr_img))


def make_dataloaders(train_pairs, val_pairs, cfg, device: torch.device) -> tuple[DataLoader, DataLoader]:
    pin_memory = device.type == 'cuda'
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


def make_grad_scaler(device: torch.device, use_amp: bool):
    if not (use_amp and device.type == 'cuda'):
        return None
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def lr_multiplier(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, cfg) -> dict[str, Any]:
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
        'val_loss': total_loss / max(1, sample_count),
        'val_psnr': total_psnr / max(1, sample_count),
        'input_psnr': total_input_psnr / max(1, sample_count),
        'delta_psnr': total_delta / max(1, sample_count),
    }


def train_one_epoch(model: nn.Module, ema_model: nn.Module, loader: DataLoader, optimizer, scaler, device: torch.device, cfg) -> dict[str, float]:
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
        'train_loss': total_loss / max(1, sample_count),
        'train_psnr': total_psnr / max(1, sample_count),
    }


def compute_image_profile(image_path: Path) -> dict[str, float]:
    image = pil_rgb(image_path).resize((64, 64), RESAMPLING_BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    return {
        'brightness': float(gray.mean()),
        'texture': float((grad_x + grad_y) * 0.5),
    }


def select_calibration_pairs(train_pairs: list[tuple[Path, Path, str]], count: int) -> list[tuple[Path, Path, str]]:
    if not train_pairs:
        return []
    if len(train_pairs) <= count:
        return train_pairs
    profiled = []
    for lr_path, hr_path, name in train_pairs:
        stats = compute_image_profile(lr_path)
        score = stats['brightness'] * 0.4 + stats['texture'] * 0.6
        profiled.append((score, (lr_path, hr_path, name)))
    profiled.sort(key=lambda item: item[0])
    selected = []
    if count == 1:
        return [profiled[len(profiled) // 2][1]]
    for index in range(count):
        position = round(index * (len(profiled) - 1) / (count - 1))
        selected.append(profiled[position][1])
    return selected


def export_calibration_dataset(train_pairs, calibration_dir: Path, eval_size: int, calibration_count: int) -> dict[str, Any]:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    selected = select_calibration_pairs(train_pairs, calibration_count)
    manifest_items = []
    for index, (lr_path, _, name) in enumerate(selected):
        image = pil_rgb(lr_path)
        if image.size != (eval_size, eval_size):
            image = ImageOps.fit(image, (eval_size, eval_size), method=RESAMPLING_BICUBIC)
        file_name = f'{index:03d}_{Path(name).stem}.png'
        out_path = calibration_dir / file_name
        image.save(out_path)
        manifest_items.append(
            {
                'index': index,
                'name': name,
                'source_lr': str(lr_path),
                'image_path': str(out_path),
                'derived_from_training': True,
            }
        )
    manifest_path = calibration_dir / 'manifest.json'
    save_json(
        manifest_path,
        {
            'count': len(manifest_items),
            'source': 'training_pairs',
            'eval_size': eval_size,
            'items': manifest_items,
        },
    )
    return {
        'calibration_dir': str(calibration_dir),
        'manifest_path': str(manifest_path),
        'count': len(manifest_items),
        'source': 'training_pairs',
        'derived_from_training': True,
    }


def audit_onnx_graph(onnx_path: Path) -> dict[str, Any]:
    if onnx is None:
        return {'checked': False, 'reason': 'onnx package unavailable', 'known_supported_or_fallback_ops': sorted(KNOWN_SUPPORTED_OR_FALLBACK_ONNX_OPS)}
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    ops = sorted({node.op_type for node in model_proto.graph.node})
    unexpected = sorted(set(ops) - KNOWN_SUPPORTED_OR_FALLBACK_ONNX_OPS)
    if unexpected:
        raise RuntimeError(f'Encountered ONNX ops not classified in the MLA support notes workflow: {unexpected}')
    return {
        'checked': True,
        'onnx_checker': 'passed',
        'ops': ops,
        'unexpected_ops': unexpected,
        'known_supported_or_fallback_ops': sorted(KNOWN_SUPPORTED_OR_FALLBACK_ONNX_OPS),
    }


def export_to_onnx(model: nn.Module, onnx_path: Path, device: torch.device, eval_size: int, verify: bool) -> dict[str, Any]:
    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, eval_size, eval_size, device=device)
    with torch.no_grad():
        export_kwargs = {
            'export_params': True,
            'opset_version': 13,
            'do_constant_folding': True,
            'input_names': ['input'],
            'output_names': ['output'],
            'dynamic_axes': None,
        }
        try:
            torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)
    metadata: dict[str, Any] = {
        'onnx_path': str(onnx_path),
        'onnx_size_kb': round(onnx_path.stat().st_size / 1024.0, 2),
    }
    metadata.update(audit_onnx_graph(onnx_path))
    if verify and ort is not None:
        cpu_dummy = dummy.detach().cpu().numpy()
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        ort_output = session.run(None, {'input': cpu_dummy})[0]
        torch_output = model(dummy).detach().cpu().numpy()
        diff = np.abs(ort_output - torch_output)
        metadata['ort_max_diff'] = float(diff.max())
        metadata['ort_mean_diff'] = float(diff.mean())
        if metadata['ort_max_diff'] >= 1e-3:
            raise RuntimeError(f"ONNX parity max diff too large: {metadata['ort_max_diff']}")
    return metadata


def build_mxq_handoff(onnx_path: Path, calibration_dir: Path, output_path: Path, cfg) -> dict[str, Any]:
    command_template = cfg.compile_command_template.strip()
    extra_args = list(cfg.compile_extra_args)
    command_preview = []
    if command_template:
        command_preview = [command_template, '--onnx', str(onnx_path), '--calibration-dir', str(calibration_dir), '--output', str(output_path)]
        command_preview.extend(extra_args)
        status = 'ready_to_execute' if cfg.run_mxq_compile else 'handoff_only'
        message = 'Compiler command template recorded in notebook config.'
    else:
        status = 'handoff_only'
        message = 'No MXQ compiler command template supplied; handoff metadata only.'
    return {
        'status': status,
        'requested_compile': cfg.run_mxq_compile,
        'message': message,
        'onnx': str(onnx_path),
        'calibration_dir': str(calibration_dir),
        'output': str(output_path),
        'output_exists': output_path.exists(),
        'command_preview': command_preview,
    }


def print_epoch_log(metrics: dict[str, Any], total_epochs: int) -> None:
    print(
        f"Epoch {metrics['epoch']:03d}/{total_epochs:03d} | "
        f"lr {metrics['learning_rate']:.2e} | "
        f"train_loss {metrics['train_loss']:.6f} | "
        f"train_psnr {metrics['train_psnr']:.4f} | "
        f"val_psnr {metrics['val_psnr']:.4f} | "
        f"baseline {metrics['input_psnr']:.4f} | "
        f"delta {metrics['delta_psnr']:+.4f} | "
        f"time {metrics['epoch_seconds']:.1f}s"
    )


def artifact_readiness(run_root: Path) -> dict[str, bool]:
    return {
        'pth_ready': (run_root / 'checkpoints' / 'best.pt').exists(),
        'onnx_ready': (run_root / 'exports' / 'best.onnx').exists(),
        'calibration_ready': (run_root / 'exports' / 'calibration' / 'manifest.json').exists(),
        'mxq_handoff_ready': (run_root / 'report.json').exists() or (run_root / 'summary.json').exists(),
    }


def localize_payload(payload: Any, local_run_root: Path) -> Any:
    if isinstance(payload, dict):
        return {key: localize_payload(value, local_run_root) for key, value in payload.items()}
    if isinstance(payload, list):
        return [localize_payload(value, local_run_root) for value in payload]
    if isinstance(payload, str):
        value = payload
        run_marker = f"{REMOTE_RUNS_ROOT}/{local_run_root.parent.name}/{local_run_root.name}"
        if run_marker in value:
            value = value.replace(run_marker, str(local_run_root))
        value = value.replace(str(REMOTE_DATA_ROOT), str(DATA_ROOT))
        value = value.replace(str(REMOTE_RUNS_ROOT), str(LOCAL_RUNS_ROOT))
        return value
    return payload

@dataclass
class ExperimentConfig:
    candidate_id: str = 'unet_experiment_1_npu_safe'
    run_mode: str = os.environ.get('UNET_EXPERIMENT_RUN_MODE', 'smoke').strip().lower()
    run_name: str = os.environ.get('UNET_EXPERIMENT_RUN_NAME', f"u_net_experiment_1_{time.strftime('%Y%m%d_%H%M%S')}")
    started_day: str = os.environ.get('UNET_EXPERIMENT_STARTED_DAY', time.strftime('%Y-%m-%d'))
    seed: int = int(os.environ.get('UNET_EXPERIMENT_SEED', '255'))
    batch_size: int = int(os.environ.get('UNET_EXPERIMENT_BATCH_SIZE', '24'))
    num_epochs: int = int(os.environ.get('UNET_EXPERIMENT_NUM_EPOCHS', '1'))
    learning_rate: float = float(os.environ.get('UNET_EXPERIMENT_LR', '3e-4'))
    weight_decay: float = float(os.environ.get('UNET_EXPERIMENT_WEIGHT_DECAY', '2e-4'))
    warmup_epochs: int = int(os.environ.get('UNET_EXPERIMENT_WARMUP_EPOCHS', '1'))
    train_patch_size: int = int(os.environ.get('UNET_EXPERIMENT_TRAIN_PATCH', '224'))
    eval_size: int = int(os.environ.get('UNET_EXPERIMENT_EVAL_SIZE', '256'))
    train_pair_limit: int | None = int(os.environ['UNET_EXPERIMENT_TRAIN_PAIR_LIMIT']) if os.environ.get('UNET_EXPERIMENT_TRAIN_PAIR_LIMIT') else 8
    val_pair_limit: int | None = int(os.environ['UNET_EXPERIMENT_VAL_PAIR_LIMIT']) if os.environ.get('UNET_EXPERIMENT_VAL_PAIR_LIMIT') else 4
    train_num_workers: int = int(os.environ.get('UNET_EXPERIMENT_TRAIN_WORKERS', '0'))
    eval_num_workers: int = int(os.environ.get('UNET_EXPERIMENT_EVAL_WORKERS', '0'))
    calibration_count: int = int(os.environ.get('UNET_EXPERIMENT_CALIBRATION_COUNT', '16'))
    ema_decay: float = float(os.environ.get('UNET_EXPERIMENT_EMA_DECAY', '0.999'))
    use_amp: bool = os.environ.get('UNET_EXPERIMENT_USE_AMP', 'true').lower() in {'1', 'true', 'yes'}
    channels_last: bool = os.environ.get('UNET_EXPERIMENT_CHANNELS_LAST', 'true').lower() in {'1', 'true', 'yes'}
    verify_onnx_export: bool = True
    run_mxq_compile: bool = os.environ.get('UNET_EXPERIMENT_RUN_MXQ_COMPILE', 'false').lower() in {'1', 'true', 'yes'}
    compile_command_template: str = os.environ.get('UNET_EXPERIMENT_MXQ_COMMAND_TEMPLATE', '')
    compile_extra_args: list[str] = field(default_factory=lambda: [arg for arg in os.environ.get('UNET_EXPERIMENT_MXQ_EXTRA_ARGS', '').split(';;') if arg])
    modal_gpu: str = os.environ.get('UNET_EXPERIMENT_MODAL_GPU', 'L40S')
    modal_timeout_minutes: int = int(os.environ.get('UNET_EXPERIMENT_MODAL_TIMEOUT_MINUTES', '120'))
    modal_data_volume: str = os.environ.get('UNET_EXPERIMENT_MODAL_DATA_VOLUME', 'lab3-data')
    modal_runs_volume: str = os.environ.get('UNET_EXPERIMENT_MODAL_RUNS_VOLUME', 'lab3-runs')
    sync_data_to_volume: bool = os.environ.get('UNET_EXPERIMENT_SYNC_DATA', 'true').lower() in {'1', 'true', 'yes'}
    force_data_sync: bool = os.environ.get('UNET_EXPERIMENT_FORCE_DATA_SYNC', 'false').lower() in {'1', 'true', 'yes'}
    execute_modal: bool = os.environ.get('UNET_EXPERIMENT_EXECUTE_MODAL', 'false').lower() in {'1', 'true', 'yes'}
    run_smoke_gate_before_full: bool = os.environ.get('UNET_EXPERIMENT_RUN_SMOKE_GATE', 'true').lower() in {'1', 'true', 'yes'}
    base_channels: int = int(os.environ.get('UNET_EXPERIMENT_BASE_CHANNELS', '56'))
    levels: int = int(os.environ.get('UNET_EXPERIMENT_LEVELS', '4'))
    slope: float = float(os.environ.get('UNET_EXPERIMENT_SLOPE', '0.10'))
    use_deep_skip_bridges: bool = os.environ.get('UNET_EXPERIMENT_USE_DEEP_SKIPS', 'false').lower() in {'1', 'true', 'yes'}

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['project_root'] = str(PROJECT_ROOT)
        payload['data_root'] = str(DATA_ROOT)
        payload['local_runs_root'] = str(LOCAL_RUNS_ROOT)
        payload['remote_data_root'] = str(REMOTE_DATA_ROOT)
        payload['remote_runs_root'] = str(REMOTE_RUNS_ROOT)
        payload['python_version'] = REMOTE_PYTHON_VERSION
        return payload


def build_config(run_mode: str | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if run_mode is not None:
        cfg.run_mode = run_mode
    if cfg.run_mode == 'full':
        cfg.num_epochs = int(os.environ.get('UNET_EXPERIMENT_NUM_EPOCHS', '34'))
        cfg.warmup_epochs = int(os.environ.get('UNET_EXPERIMENT_WARMUP_EPOCHS', '5'))
        cfg.train_pair_limit = int(os.environ['UNET_EXPERIMENT_TRAIN_PAIR_LIMIT']) if os.environ.get('UNET_EXPERIMENT_TRAIN_PAIR_LIMIT') else None
        cfg.val_pair_limit = int(os.environ['UNET_EXPERIMENT_VAL_PAIR_LIMIT']) if os.environ.get('UNET_EXPERIMENT_VAL_PAIR_LIMIT') else None
        cfg.calibration_count = int(os.environ.get('UNET_EXPERIMENT_CALIBRATION_COUNT', '128'))
    return cfg

def run_remote_pipeline(cfg_payload: dict[str, Any], data_mount_root: str, runs_mount_root: str, runs_volume) -> dict[str, Any]:
    cfg = ExperimentConfig(**{key: value for key, value in cfg_payload.items() if key in ExperimentConfig.__dataclass_fields__})
    set_seed(cfg.seed)
    remote_data_root = Path(data_mount_root) / 'Data'
    remote_runs_root = Path(runs_mount_root) / 'runs'
    device = resolve_device()

    pairing_audit = run_pairing_audit(remote_data_root)
    train_pairs = collect_train_pairs(remote_data_root, cfg.train_pair_limit)
    val_pairs = collect_val_pairs(remote_data_root, cfg.val_pair_limit)
    if not train_pairs:
        raise FileNotFoundError(f'No training pairs found under {remote_data_root}')
    if not val_pairs:
        raise FileNotFoundError(f'No validation pairs found under {remote_data_root}')

    run_root = remote_runs_root / cfg.started_day / cfg.run_name
    checkpoint_dir = run_root / 'checkpoints'
    export_dir = run_root / 'exports'
    calibration_dir = export_dir / 'calibration'
    for path in [run_root, checkpoint_dir, export_dir, calibration_dir]:
        path.mkdir(parents=True, exist_ok=True)

    run_config_payload = cfg.as_json()
    run_config_payload.update(
        {
            'data_root': str(remote_data_root),
            'run_root': str(run_root),
            'backend': 'modal',
            'remote_data_root': str(remote_data_root),
            'remote_runs_root': str(remote_runs_root),
        }
    )
    save_json(run_root / 'run_config.json', run_config_payload)
    runs_volume.commit()

    train_loader, val_loader = make_dataloaders(train_pairs, val_pairs, cfg, device)
    model = SingleBridgeUNet(
        base_channels=cfg.base_channels,
        levels=cfg.levels,
        slope=cfg.slope,
        use_deep_skip_bridges=cfg.use_deep_skip_bridges,
    ).to(device)
    if cfg.channels_last and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    contract = verify_model_contract(model, cfg.eval_size)
    verify_residual_l1_batch(model, train_loader, device, cfg.channels_last)
    print('Remote model contract:')
    print(json.dumps(contract, indent=2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_multiplier(epoch, cfg.num_epochs, cfg.warmup_epochs),
    )
    scaler = make_grad_scaler(device, cfg.use_amp)
    ema_model = create_ema_model(model)
    best_checkpoint = checkpoint_dir / 'best.pt'
    last_checkpoint = checkpoint_dir / 'last.pt'
    best_val_psnr = float('-inf')
    best_epoch = 0
    latest_metrics: dict[str, Any] = {}

    for epoch in range(cfg.num_epochs):
        epoch_index = epoch + 1
        start_time = time.time()
        train_metrics = train_one_epoch(model, ema_model, train_loader, optimizer, scaler, device, cfg)
        eval_metrics = evaluate_loader(ema_model, val_loader, device, cfg)
        scheduler.step()
        latest_metrics = {
            'epoch': epoch_index,
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'epoch_seconds': time.time() - start_time,
            **train_metrics,
            **eval_metrics,
        }
        append_jsonl(run_root / 'metrics.jsonl', latest_metrics)
        latest_status = {
            'phase': 'training',
            'epoch': epoch_index,
            'best_epoch': best_epoch,
            'best_val_psnr': best_val_psnr,
            'latest_metrics': latest_metrics,
        }
        save_json(run_root / 'latest_status.json', latest_status)
        torch.save({'epoch': epoch_index, 'model_state_dict': ema_model.state_dict(), 'config': run_config_payload}, last_checkpoint)
        if eval_metrics['val_psnr'] > best_val_psnr:
            best_val_psnr = float(eval_metrics['val_psnr'])
            best_epoch = epoch_index
            torch.save({'epoch': epoch_index, 'model_state_dict': ema_model.state_dict(), 'config': run_config_payload}, best_checkpoint)
            print(f'New best checkpoint at epoch {epoch_index}: {best_val_psnr:.4f} dB')
        print_epoch_log(latest_metrics, cfg.num_epochs)
        runs_volume.commit()

    eval_model = SingleBridgeUNet(
        base_channels=cfg.base_channels,
        levels=cfg.levels,
        slope=cfg.slope,
        use_deep_skip_bridges=cfg.use_deep_skip_bridges,
    ).to(device)
    best_state = torch.load(best_checkpoint, map_location=device)
    eval_model.load_state_dict(best_state['model_state_dict'])
    if cfg.channels_last and device.type == 'cuda':
        eval_model = eval_model.to(memory_format=torch.channels_last)
    evaluation = evaluate_loader(eval_model, val_loader, device, cfg)

    onnx_path = export_dir / 'best.onnx'
    onnx_summary = export_to_onnx(eval_model, onnx_path, device, cfg.eval_size, cfg.verify_onnx_export)
    print('ONNX export complete.')
    print(json.dumps(onnx_summary, indent=2))

    calibration_summary = export_calibration_dataset(train_pairs, calibration_dir, cfg.eval_size, cfg.calibration_count)
    print('Calibration export complete.')
    print(json.dumps(calibration_summary, indent=2))

    mxq_summary = build_mxq_handoff(onnx_path, calibration_dir, export_dir / 'best.mxq', cfg)
    print('MXQ handoff status:')
    print(json.dumps(mxq_summary, indent=2))

    gates = {
        'contract_pass': contract['input_shape'] == [1, 3, 256, 256] and contract['output_shape'] == [1, 3, 256, 256],
        'leaf_module_pass': contract['leaf_audit']['all_leaf_modules_known'],
        'onnx_op_pass': not onnx_summary.get('unexpected_ops'),
        'onnx_pass': onnx_summary.get('onnx_checker') == 'passed' and onnx_summary.get('ort_max_diff', 0.0) < 1e-3,
        'calibration_pass': calibration_summary['derived_from_training'] and calibration_summary['count'] > 0,
        'mxq_handoff_pass': mxq_summary['status'] in {'handoff_only', 'ready_to_execute', 'completed'},
    }
    gates['promotion_pass'] = all(gates.values())

    summary = {
        'backend': 'modal',
        'candidate': {
            'candidate_id': cfg.candidate_id,
            'architecture': 'u_net_single_bridge_nobn',
            'base_channels': cfg.base_channels,
            'levels': cfg.levels,
            'use_deep_skip_bridges': cfg.use_deep_skip_bridges,
            'slope': cfg.slope,
        },
        'config': run_config_payload,
        'pairing_audit': pairing_audit,
        'pair_summary': {
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'train_preview': [name for _, _, name in train_pairs[:3]],
            'val_preview': [name for _, _, name in val_pairs[:3]],
        },
        'device': str(device),
        'run_day': cfg.started_day,
        'run_root': str(run_root),
        'best_checkpoint': str(best_checkpoint),
        'onnx_path': str(onnx_path),
        'mxq_path': str(export_dir / 'best.mxq'),
        'summary_path': str(run_root / 'summary.json'),
        'report_path': str(run_root / 'report.json'),
        'model_contract': contract,
        'training': {
            'best_epoch': best_epoch,
            'best_val_psnr': best_val_psnr,
            'latest_metrics': latest_metrics,
            'best_checkpoint': str(best_checkpoint),
            'last_checkpoint': str(last_checkpoint),
        },
        'evaluation': evaluation,
        'onnx': onnx_summary,
        'calibration': calibration_summary,
        'mxq': mxq_summary,
        'gates': gates,
    }
    save_json(run_root / 'summary.json', {'summary': summary})
    save_json(
        run_root / 'latest_status.json',
        {
            'phase': 'pipeline_complete',
            'epoch': best_epoch,
            'best_epoch': best_epoch,
            'best_val_psnr': best_val_psnr,
            'latest_metrics': latest_metrics,
            'summary_path': str(run_root / 'summary.json'),
            'report_path': str(run_root / 'report.json'),
            'promotion_pass': gates['promotion_pass'],
        },
    )
    runs_volume.commit()
    summary['artifact_readiness'] = artifact_readiness(run_root)
    save_json(run_root / 'report.json', summary)
    runs_volume.commit()
    return summary


# Built once at import so Modal decorators see GPU/volume names from env-backed config.
cfg = build_config()

DATA_VOLUME = modal.Volume.from_name(cfg.modal_data_volume, create_if_missing=True)
RUNS_VOLUME = modal.Volume.from_name(cfg.modal_runs_volume, create_if_missing=True)
APP = modal.App('lab3-unet-experiment-1')
IMAGE = (
    modal.Image.debian_slim(python_version=REMOTE_PYTHON_VERSION)
    .pip_install(
        'torch==2.10.0',
        'onnx==1.20.1',
        'onnxruntime==1.24.1',
        'numpy==2.4.2',
        'pillow==12.1.1',
    )
)


@APP.function(
    image=IMAGE,
    gpu=cfg.modal_gpu,
    timeout=cfg.modal_timeout_minutes * 60,
    volumes={str(REMOTE_DATA_MOUNT): DATA_VOLUME, str(REMOTE_RUNS_MOUNT): RUNS_VOLUME},
)
def run_unet_experiment_1(cfg_payload):
    return run_remote_pipeline(cfg_payload, str(REMOTE_DATA_MOUNT), str(REMOTE_RUNS_MOUNT), RUNS_VOLUME)


def upload_required_data(force: bool) -> dict[str, Any]:
    uploaded = []
    cleaned = []
    if not cfg.sync_data_to_volume:
        return {'status': 'skipped', 'uploaded': uploaded, 'cleaned': cleaned, 'volume': cfg.modal_data_volume}
    for name in ['HR_train', 'LR_train', 'HR_val', 'LR_val']:
        remote_dir = f'/Data/{name}'
        try:
            DATA_VOLUME.remove_file(remote_dir, recursive=True)
            cleaned.append(remote_dir)
        except Exception:
            pass
    with DATA_VOLUME.batch_upload(force=True) as batch:
        for name in ['HR_train', 'LR_train', 'HR_val', 'LR_val']:
            local_dir = DATA_ROOT / name
            if not local_dir.exists():
                raise FileNotFoundError(f'Missing local data directory: {local_dir}')
            batch.put_directory(str(local_dir), f'/Data/{name}')
            uploaded.append({'local': str(local_dir), 'remote': f'/Data/{name}'})
    return {
        'status': 'uploaded' if force else 'refreshed_required_training_data',
        'uploaded': uploaded,
        'cleaned': cleaned,
        'volume': cfg.modal_data_volume,
        'note': 'Required training and validation folders were uploaded for this run to avoid stale volume state.',
    }


def read_volume_file(volume, remote_path: str) -> bytes:
    buffer = bytearray()
    for chunk in volume.read_file(remote_path):
        buffer.extend(chunk)
    return bytes(buffer)


def sync_run_from_volume(run_day: str, run_name: str) -> Path:
    remote_prefix = f'runs/{run_day}/{run_name}'
    local_run_root = LOCAL_RUNS_ROOT / run_day / run_name
    local_run_root.mkdir(parents=True, exist_ok=True)
    for entry in RUNS_VOLUME.listdir(remote_prefix, recursive=True):
        remote_path = getattr(entry, 'path', None)
        if not remote_path:
            continue
        normalized_remote = remote_path.lstrip('/')
        try:
            payload = read_volume_file(RUNS_VOLUME, normalized_remote)
        except Exception:
            continue
        relative = Path(normalized_remote).relative_to(remote_prefix)
        local_path = local_run_root / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(payload)
    for json_name in ['run_config.json', 'summary.json', 'report.json', 'latest_status.json']:
        json_path = local_run_root / json_name
        if not json_path.exists():
            continue
        payload = json.loads(json_path.read_text(encoding='utf-8'))
        localized = localize_payload(payload, local_run_root)
        json_path.write_text(json.dumps(localized, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return local_run_root


def launch_remote(cfg_for_run: ExperimentConfig) -> dict[str, Any]:
    print('Uploading required data to Modal volume...')
    upload_result = upload_required_data(force=cfg_for_run.force_data_sync)
    print(json.dumps(upload_result, indent=2))
    # Do not nest `APP.run()` here: `modal run` invokes a local_entrypoint with the app already running.
    with modal.enable_output():
        remote_summary = run_unet_experiment_1.remote(cfg_for_run.as_json())
    local_run_root = sync_run_from_volume(cfg_for_run.started_day, cfg_for_run.run_name)
    report_path = local_run_root / 'report.json'
    if report_path.exists():
        local_summary = json.loads(report_path.read_text(encoding='utf-8'))
    else:
        local_summary = localize_payload(remote_summary, local_run_root)
    local_summary['synced_local_run_root'] = str(local_run_root)
    local_summary['artifact_readiness'] = artifact_readiness(local_run_root)
    return local_summary


def make_smoke_cfg(base_cfg: ExperimentConfig) -> ExperimentConfig:
    smoke_cfg = copy.deepcopy(base_cfg)
    smoke_cfg.run_mode = 'smoke'
    smoke_cfg.run_name = f"{base_cfg.run_name}_smoke_gate"
    smoke_cfg.num_epochs = 1
    smoke_cfg.warmup_epochs = 1
    smoke_cfg.train_pair_limit = 8
    smoke_cfg.val_pair_limit = 4
    smoke_cfg.calibration_count = min(base_cfg.calibration_count, 16)
    return smoke_cfg


def run_cli() -> None:
    cfg = build_config()
    pairing_audit = run_pairing_audit(DATA_ROOT)
    print("Pairing audit passed.")
    print(json.dumps(pairing_audit, indent=2))
    print("Notebook and Modal configuration:")
    print(json.dumps(cfg.as_json(), indent=2))
    if not cfg.execute_modal:
        print("Modal execution is disabled. Set UNET_EXPERIMENT_EXECUTE_MODAL=true to launch the remote app.")
        print("Pairing audit and config preflight completed locally.")
        return
    final_summary = None
    if cfg.run_mode == "full" and cfg.run_smoke_gate_before_full:
        smoke_cfg = make_smoke_cfg(cfg)
        print("Running remote smoke gate before full experiment...")
        smoke_summary = launch_remote(smoke_cfg)
        print("Smoke summary:")
        print(json.dumps(smoke_summary, indent=2))
        smoke_gates = smoke_summary.get("gates", {})
        if not all(
            [
                smoke_gates.get("contract_pass"),
                smoke_gates.get("leaf_module_pass"),
                smoke_gates.get("onnx_op_pass"),
                smoke_gates.get("onnx_pass"),
                smoke_gates.get("calibration_pass"),
                smoke_gates.get("mxq_handoff_pass"),
            ]
        ):
            raise RuntimeError(
                "Smoke gate failed. Full run aborted to avoid spending the 34-epoch budget."
            )
    print(f"Launching remote run mode={cfg.run_mode} name={cfg.run_name}")
    final_summary = launch_remote(cfg)
    print("Final synced summary:")
    print(json.dumps(final_summary, indent=2))
    ap = final_summary.get("artifact_readiness") or final_summary.get("artifact_summary_payload")
    print("artifact_readiness:", json.dumps(ap, indent=2) if ap else "n/a")


@APP.local_entrypoint()
def main() -> None:
    run_cli()


if __name__ == "__main__":
    with modal.enable_output():
        with APP.run():
            run_cli()
