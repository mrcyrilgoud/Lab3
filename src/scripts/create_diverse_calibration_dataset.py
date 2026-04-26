#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "DataSampling" / "L3_calibration_diverse"

RESAMPLING_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


@dataclass(frozen=True)
class PairRecord:
    split: str
    stem: str
    lr_path: Path
    hr_path: Path


@dataclass(frozen=True)
class FeatureRecord:
    pair: PairRecord
    brightness: float
    contrast: float
    texture: float
    saturation: float
    diversity_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a diverse calibration dataset from Lab 3 training pairs."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--count",
        type=int,
        default=256,
        help="Number of calibration samples to export (default: 256).",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=256,
        help="Output image size (must remain 256 for Lab 3 contract).",
    )
    parser.add_argument("--seed", type=int, default=255)
    return parser.parse_args()


def _collect_images(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _index_by_stem(paths: list[Path], label: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in paths:
        key = path.stem
        if key in mapping:
            duplicates.append(key)
            continue
        mapping[key] = path
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate {label} stems found: {preview}")
    return mapping


def collect_train_pairs(data_root: Path) -> list[PairRecord]:
    records: list[PairRecord] = []
    for index in range(1, 5):
        split = f"train{index}"
        lr_root = data_root / "LR_train" / f"LR_train{index}"
        hr_root = data_root / "HR_train" / f"HR_train{index}"
        if not lr_root.is_dir() or not hr_root.is_dir():
            raise FileNotFoundError(f"Missing training split directories: {lr_root} and/or {hr_root}")

        lr_map = _index_by_stem(_collect_images(lr_root), f"LR {split}")
        hr_map = _index_by_stem(_collect_images(hr_root), f"HR {split}")
        shared = sorted(set(lr_map) & set(hr_map))
        if not shared:
            raise FileNotFoundError(f"No paired samples found in split {split}")

        missing_lr = sorted(set(hr_map) - set(lr_map))
        missing_hr = sorted(set(lr_map) - set(hr_map))
        if missing_lr or missing_hr:
            raise ValueError(
                f"Split {split} has basename mismatch: missing_lr={len(missing_lr)} missing_hr={len(missing_hr)}"
            )

        for stem in shared:
            records.append(
                PairRecord(
                    split=split,
                    stem=stem,
                    lr_path=lr_map[stem],
                    hr_path=hr_map[stem],
                )
            )
    if not records:
        raise RuntimeError("No training pairs discovered.")
    return records


def compute_features(pair: PairRecord) -> FeatureRecord:
    with Image.open(pair.lr_path) as raw:
        rgb = raw.convert("RGB")
        if rgb.size != (64, 64):
            rgb = ImageOps.fit(rgb, (64, 64), method=RESAMPLING_BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0

    gray = arr.mean(axis=2)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    grad_y = float(np.abs(np.diff(gray, axis=0)).mean())
    grad_x = float(np.abs(np.diff(gray, axis=1)).mean())
    texture = 0.5 * (grad_x + grad_y)
    saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    diversity_score = 0.35 * contrast + 0.35 * texture + 0.20 * saturation + 0.10 * abs(brightness - 0.5)

    return FeatureRecord(
        pair=pair,
        brightness=brightness,
        contrast=contrast,
        texture=texture,
        saturation=saturation,
        diversity_score=diversity_score,
    )


def quantile_edges(values: list[float], bins: int) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return [0.0, 1.0]
    edges: list[float] = []
    for q in range(1, bins):
        idx = round((len(ordered) - 1) * (q / bins))
        edges.append(ordered[idx])
    return edges


def bucket_index(value: float, edges: list[float]) -> int:
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return len(edges)


def select_diverse_subset(records: list[FeatureRecord], target_count: int, seed: int) -> list[FeatureRecord]:
    if target_count <= 0:
        raise ValueError("--count must be > 0")
    if len(records) <= target_count:
        return records

    # Two-dimensional stratification keeps both brightness and texture varied.
    brightness_edges = quantile_edges([r.brightness for r in records], bins=4)
    texture_edges = quantile_edges([r.texture for r in records], bins=4)
    buckets: dict[tuple[int, int], list[FeatureRecord]] = defaultdict(list)
    for record in records:
        b_idx = bucket_index(record.brightness, brightness_edges)
        t_idx = bucket_index(record.texture, texture_edges)
        buckets[(b_idx, t_idx)].append(record)

    rng = random.Random(seed)
    for items in buckets.values():
        rng.shuffle(items)
        items.sort(key=lambda item: item.diversity_score, reverse=True)

    # Round-robin over buckets for broad coverage and split balancing.
    selected: list[FeatureRecord] = []
    used_stems: set[str] = set()
    ordered_keys = sorted(buckets.keys(), key=lambda key: (-len(buckets[key]), key))
    while len(selected) < target_count:
        progress = False
        for key in ordered_keys:
            items = buckets[key]
            while items and items[0].pair.stem in used_stems:
                items.pop(0)
            if not items:
                continue
            choice = items.pop(0)
            selected.append(choice)
            used_stems.add(choice.pair.stem)
            progress = True
            if len(selected) >= target_count:
                break
        if not progress:
            break

    if len(selected) < target_count:
        leftovers = [r for r in records if r.pair.stem not in used_stems]
        leftovers.sort(key=lambda item: item.diversity_score, reverse=True)
        selected.extend(leftovers[: max(0, target_count - len(selected))])

    return selected[:target_count]


def export_images(
    selected: list[FeatureRecord],
    output_dir: Path,
    eval_size: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket_distribution: dict[str, dict[str, int]] = {"split_counts": {}, "brightness_bin_counts": {}, "texture_bin_counts": {}}
    split_counts: Counter[str] = Counter()
    brightness_bins: Counter[str] = Counter()
    texture_bins: Counter[str] = Counter()

    brightness_edges = quantile_edges([r.brightness for r in selected], bins=4)
    texture_edges = quantile_edges([r.texture for r in selected], bins=4)

    manifest_items: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        file_name = f"{index:04d}_{record.pair.split}_{record.pair.stem}.png"
        lr_out = output_dir / file_name

        with Image.open(record.pair.lr_path) as lr_raw:
            lr_rgb = lr_raw.convert("RGB")
            if lr_rgb.size != (eval_size, eval_size):
                lr_rgb = ImageOps.fit(lr_rgb, (eval_size, eval_size), method=RESAMPLING_BICUBIC)
            lr_rgb.save(lr_out)

        split_counts[record.pair.split] += 1
        brightness_bin = f"b{bucket_index(record.brightness, brightness_edges)}"
        texture_bin = f"t{bucket_index(record.texture, texture_edges)}"
        brightness_bins[brightness_bin] += 1
        texture_bins[texture_bin] += 1

        manifest_items.append(
            {
                "index": index,
                "name": file_name,
                "split": record.pair.split,
                "stem": record.pair.stem,
                "source_lr": str(record.pair.lr_path),
                "source_hr": str(record.pair.hr_path),
                "exported_lr": str(lr_out),
                "brightness": round(record.brightness, 6),
                "contrast": round(record.contrast, 6),
                "texture": round(record.texture, 6),
                "saturation": round(record.saturation, 6),
                "diversity_score": round(record.diversity_score, 6),
                "brightness_bin": brightness_bin,
                "texture_bin": texture_bin,
                "derived_from_training": True,
            }
        )

    bucket_distribution["split_counts"] = dict(sorted(split_counts.items()))
    bucket_distribution["brightness_bin_counts"] = dict(sorted(brightness_bins.items()))
    bucket_distribution["texture_bin_counts"] = dict(sorted(texture_bins.items()))
    return manifest_items, bucket_distribution


def write_manifest(
    *,
    output_dir: Path,
    count: int,
    eval_size: int,
    seed: int,
    items: list[dict[str, Any]],
    train_total: int,
    distributions: dict[str, dict[str, int]],
) -> None:
    payload = {
        "count": count,
        "eval_size": eval_size,
        "seed": seed,
        "source": "training_pairs",
        "input_domain": "lr",
        "derived_from_training": True,
        "selection": {
            "strategy": "quantile_stratified_brightness_texture_with_diversity_round_robin",
            "train_pairs_total": train_total,
        },
        "distribution": distributions,
        "items": items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.eval_size != 256:
        raise ValueError("Lab 3 requires a 256x256 input/output contract; pass --eval-size 256.")
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    train_pairs = collect_train_pairs(data_root)
    features = [compute_features(pair) for pair in train_pairs]
    selected = select_diverse_subset(features, target_count=args.count, seed=args.seed)

    if output_dir.exists():
        for child in output_dir.glob("*"):
            if child.is_dir():
                for nested in child.glob("*"):
                    nested.unlink()
                child.rmdir()
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    items, distributions = export_images(selected, output_dir=output_dir, eval_size=args.eval_size)
    write_manifest(
        output_dir=output_dir,
        count=len(items),
        eval_size=args.eval_size,
        seed=args.seed,
        items=items,
        train_total=len(train_pairs),
        distributions=distributions,
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "manifest_path": str(output_dir / "manifest.json"),
                "count": len(items),
                "train_pairs_total": len(train_pairs),
                "derived_from_training": True,
                "distribution": distributions,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
