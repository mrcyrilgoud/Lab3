#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
EXPECTED_TRAIN_PAIRS = 3036
EXPECTED_VAL_PAIRS = 100

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINES_DIR = PROJECT_ROOT / "src" / "pipelines"
RUN_MODAL_EXPERIMENT_PATH = PROJECT_ROOT / "src" / "scripts" / "run_modal_experiment.py"
MODAL_APP_PATH = PROJECT_ROOT / "src" / "scripts" / "lab3_modal_app.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Lab 3 HR/LR pairing and current Modal pipeline data assumptions.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Data")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for the JSON audit report.")
    return parser.parse_args()


def import_pipeline_lib() -> Any:
    if str(PIPELINES_DIR) not in sys.path:
        sys.path.insert(0, str(PIPELINES_DIR))
    import lab3_pipeline_lib  # type: ignore

    return lab3_pipeline_lib


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compute_psnr_arrays(lr_img: Image.Image, hr_img: Image.Image) -> float:
    lr = np.asarray(lr_img, dtype=np.float32) / 255.0
    hr = np.asarray(hr_img, dtype=np.float32) / 255.0
    mse = float(np.mean((lr - hr) ** 2))
    mse = max(mse, 1e-12)
    return -10.0 * math.log10(mse)


def file_maps(directory: Path) -> tuple[dict[str, Path], list[str]]:
    if not directory.exists():
        return {}, []
    files: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in files:
            duplicates.append(path.stem)
            continue
        files[path.stem] = path
    return files, duplicates


def audit_split(hr_dir: Path, lr_dir: Path, split_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": split_name,
        "hr_dir": str(hr_dir),
        "lr_dir": str(lr_dir),
        "exists": hr_dir.exists() and lr_dir.exists(),
    }
    if not hr_dir.exists():
        result["error"] = f"Missing HR directory: {hr_dir}"
        result["passed"] = False
        return result
    if not lr_dir.exists():
        result["error"] = f"Missing LR directory: {lr_dir}"
        result["passed"] = False
        return result

    hr_map, hr_duplicates = file_maps(hr_dir)
    lr_map, lr_duplicates = file_maps(lr_dir)
    shared = sorted(set(hr_map) & set(lr_map))
    hr_only = sorted(set(hr_map) - set(lr_map))
    lr_only = sorted(set(lr_map) - set(hr_map))

    unreadable: list[dict[str, Any]] = []
    size_mismatches: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    for stem in shared:
        try:
            with Image.open(hr_map[stem]) as hr_raw, Image.open(lr_map[stem]) as lr_raw:
                hr_img = hr_raw.convert("RGB")
                lr_img = lr_raw.convert("RGB")
                if hr_img.size != lr_img.size:
                    size_mismatches.append(
                        {
                            "stem": stem,
                            "hr_size": list(hr_img.size),
                            "lr_size": list(lr_img.size),
                        }
                    )
                psnr_values.append(compute_psnr_arrays(lr_img, hr_img))
        except Exception as exc:  # pragma: no cover - defensive audit
            unreadable.append({"stem": stem, "error": str(exc)})

    result.update(
        {
            "hr_count": len(hr_map),
            "lr_count": len(lr_map),
            "paired_count": len(shared),
            "hr_only_count": len(hr_only),
            "lr_only_count": len(lr_only),
            "duplicate_hr_basenames": hr_duplicates,
            "duplicate_lr_basenames": lr_duplicates,
            "hr_only_examples": hr_only[:10],
            "lr_only_examples": lr_only[:10],
            "unreadable_pairs": unreadable[:10],
            "size_mismatches": size_mismatches[:10],
            "name_preview": shared[:5],
            "psnr": {
                "count": len(psnr_values),
                "mean": round(statistics.fmean(psnr_values), 3) if psnr_values else None,
                "median": round(statistics.median(psnr_values), 3) if psnr_values else None,
                "p10": round(percentile(psnr_values, 0.10), 3) if psnr_values else None,
                "p90": round(percentile(psnr_values, 0.90), 3) if psnr_values else None,
            },
        }
    )
    result["passed"] = not any(
        [
            hr_duplicates,
            lr_duplicates,
            hr_only,
            lr_only,
            unreadable,
            size_mismatches,
            len(shared) == 0,
        ]
    )
    return result


def train_split_dirs(data_root: Path) -> list[tuple[str, Path, Path]]:
    items: list[tuple[str, Path, Path]] = []
    for index in range(1, 5):
        name = f"HR_train{index}"
        items.append(
            (
                name,
                data_root / "HR_train" / name,
                data_root / "LR_train" / f"LR_train{index}",
            )
        )
    return items


def pair_name_distribution(pairs: list[tuple[Path, Path, str]], limit: int | None = None) -> dict[str, int]:
    subset = pairs if limit is None else pairs[:limit]
    counts: Counter[str] = Counter()
    for _, _, name in subset:
        split = name.split("/", 1)[0] if "/" in name else "val"
        counts[split] += 1
    return dict(sorted(counts.items()))


def discover_notebook_paths() -> dict[str, Any]:
    canonical_from_launcher = PROJECT_ROOT / "src" / "lab3_wide_residual_nobn_modal_app.ipynb"
    repo_root_canonical = PROJECT_ROOT / "lab3_wide_residual_nobn_modal_app.ipynb"
    baseline_notebook = PROJECT_ROOT / "experiments" / "00_baseline" / "lab3_wide_residual_nobn_modal_app.ipynb"
    return {
        "launcher_expected": {"path": str(canonical_from_launcher), "exists": canonical_from_launcher.exists()},
        "repo_root_expected": {"path": str(repo_root_canonical), "exists": repo_root_canonical.exists()},
        "baseline_notebook": {"path": str(baseline_notebook), "exists": baseline_notebook.exists()},
    }


def build_code_audit(lib: Any, train_pairs: list[tuple[Path, Path, str]], val_pairs: list[tuple[Path, Path, str]]) -> dict[str, Any]:
    run_pipeline_source = inspect.getsource(lib.run_pipeline)
    getitem_source = inspect.getsource(lib.PairedSRDataset.__getitem__)
    random_crop_source = inspect.getsource(lib.random_crop_pair)
    run_modal_source = read_text(RUN_MODAL_EXPERIMENT_PATH)
    modal_app_source = read_text(MODAL_APP_PATH)

    train_names = [name for _, _, name in train_pairs]
    val_names = [name for _, _, name in val_pairs]
    train_stem_collisions: dict[str, list[str]] = defaultdict(list)
    for _, _, name in train_pairs:
        split, stem = name.split("/", 1)
        train_stem_collisions[stem].append(split)

    repeated_train_stems = {
        stem: sorted(splits)
        for stem, splits in train_stem_collisions.items()
        if len(set(splits)) > 1
    }

    slice_examples = {
        "train8": pair_name_distribution(train_pairs, limit=8),
        "train64": pair_name_distribution(train_pairs, limit=64),
        "train256": pair_name_distribution(train_pairs, limit=256),
        "train1024": pair_name_distribution(train_pairs, limit=1024),
        "full_train": pair_name_distribution(train_pairs, limit=None),
        "val4_preview": val_names[:4],
        "val16_preview": val_names[:16],
    }

    return {
        "pipeline_lib_path": str(Path(inspect.getsourcefile(lib)).resolve()),
        "launcher_paths": discover_notebook_paths(),
        "run_pipeline_calls_pairing_audit": "run_pairing_audit(" in run_pipeline_source,
        "train_pairs_collector": {
            "path": str(Path(inspect.getsourcefile(lib.collect_train_pairs)).resolve()),
            "uses_structured_split_layout": 'data_root / "LR_train"' in inspect.getsource(lib.collect_train_pairs),
            "allows_flat_fallback_layout": 'data_root / "train" / "LR"' in inspect.getsource(lib.collect_train_pairs),
        },
        "val_pairs_collector": {
            "path": str(Path(inspect.getsourcefile(lib.collect_val_pairs)).resolve()),
            "search_order": [
                "Data/LR_val <-> Data/HR_val",
                "Data/val/LR_val <-> Data/val/HR_val",
                "Data/val/LR <-> Data/val/HR",
            ],
            "allows_fallback_layouts": 'data_root / "val" / "LR"' in inspect.getsource(lib.collect_val_pairs),
        },
        "dataset_sampling": {
            "fixed_tuple_indexing": "lr_path, hr_path, name = self.pairs[index]" in getitem_source,
            "shared_rng_for_train_crop_and_aug": "random_crop_pair(lr_img, hr_img, self.patch_size, rng)" in getitem_source
            and "augment_pair(lr_img, hr_img, rng)" in getitem_source,
            "train_seed_formula": "self.seed + index" if "self.seed + index" in getitem_source else "unknown",
            "epoch_aware_sampling": hasattr(lib.PairedSRDataset, "set_epoch") or "epoch" in getitem_source,
            "val_resize_applied_to_both_sides": getitem_source.count("ImageOps.fit(") >= 2
            and "(self.eval_size, self.eval_size)" in getitem_source,
            "random_crop_uses_one_box": "box = (left, top, left + size, top + size)" in random_crop_source,
        },
        "naming_and_indexing": {
            "train_pair_name_unique": len(train_names) == len(set(train_names)),
            "val_pair_name_unique": len(val_names) == len(set(val_names)),
            "reused_train_stems_across_splits": len(repeated_train_stems),
            "reused_train_stem_examples": dict(list(sorted(repeated_train_stems.items()))[:10]),
            "slice_examples": slice_examples,
        },
        "modal_sync": {
            "launcher_expected_notebook_exists": 'CANONICAL_NOTEBOOK = PROJECT_ROOT / "lab3_wide_residual_nobn_modal_app.ipynb"' in run_modal_source
            and (PROJECT_ROOT / "src" / "lab3_wide_residual_nobn_modal_app.ipynb").exists(),
            "volume_sync_can_skip_existing_data": 'if not force and volume_path_exists(volume_name, "/Data")' in modal_app_source,
            "uses_modal_cli_for_volume_ops": '"modal", "volume", "put"' in modal_app_source
            or '"modal", "volume", "get"' in modal_app_source,
        },
    }


def build_pairing_audit(data_root: Path, lib: Any) -> dict[str, Any]:
    train_splits = [audit_split(hr_dir, lr_dir, split_name) for split_name, hr_dir, lr_dir in train_split_dirs(data_root)]
    val_split = audit_split(data_root / "HR_val", data_root / "LR_val", "val")
    train_pairs = lib.collect_train_pairs(data_root)
    val_pairs = lib.collect_val_pairs(data_root)

    weighted_sum = 0.0
    weighted_count = 0
    for split in train_splits:
        count = int(split["psnr"]["count"] or 0)
        mean = split["psnr"]["mean"]
        if count and mean is not None:
            weighted_sum += float(mean) * count
            weighted_count += count
    overall_train_mean = round(weighted_sum / weighted_count, 3) if weighted_count else None
    val_mean = val_split["psnr"]["mean"]
    train_val_gap = round(overall_train_mean - val_mean, 3) if overall_train_mean is not None and val_mean is not None else None

    expected_ok = len(train_pairs) == EXPECTED_TRAIN_PAIRS and len(val_pairs) == EXPECTED_VAL_PAIRS
    passed = all(item.get("passed") for item in train_splits) and val_split.get("passed") and expected_ok
    return {
        "data_root": str(data_root),
        "expected": {
            "train_pairs": EXPECTED_TRAIN_PAIRS,
            "val_pairs": EXPECTED_VAL_PAIRS,
        },
        "observed": {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
        },
        "expected_totals_match": expected_ok,
        "train_splits": train_splits,
        "val_split": val_split,
        "train_val_baseline_psnr_gap_db": train_val_gap,
        "passed": passed,
    }


def classify_issue(pairing_audit: dict[str, Any], code_audit: dict[str, Any]) -> dict[str, Any]:
    split_failures = [
        split["split"]
        for split in pairing_audit["train_splits"] + [pairing_audit["val_split"]]
        if not split.get("passed")
    ]
    if split_failures:
        return {
            "main_issue": "data_pairing",
            "reason": f"Integrity failures were found in splits: {split_failures}.",
        }

    gap = pairing_audit.get("train_val_baseline_psnr_gap_db")
    epoch_aware = code_audit["dataset_sampling"]["epoch_aware_sampling"]
    audit_gate = code_audit["run_pipeline_calls_pairing_audit"]
    if gap is not None and gap >= 5.0:
        return {
            "main_issue": "something_else",
            "reason": "On-disk pairs are clean; the dominant signal is the train/val difficulty gap, with pipeline guardrail gaps as secondary risk.",
            "secondary_risks": {
                "epoch_aware_sampling_missing": not epoch_aware,
                "pairing_audit_not_enforced_in_run_pipeline": not audit_gate,
                "slice_bias_for_limited_runs": code_audit["naming_and_indexing"]["slice_examples"]["train256"] == {"HR_train1": 256},
            },
        }
    return {
        "main_issue": "preprocessing",
        "reason": "Pairing is clean, but the pipeline still needs stronger preprocessing/audit guardrails.",
        "secondary_risks": {
            "epoch_aware_sampling_missing": not epoch_aware,
            "pairing_audit_not_enforced_in_run_pipeline": not audit_gate,
        },
    }


def build_report(data_root: Path) -> dict[str, Any]:
    lib = import_pipeline_lib()
    pairing_audit = build_pairing_audit(data_root, lib)
    train_pairs = lib.collect_train_pairs(data_root)
    val_pairs = lib.collect_val_pairs(data_root)
    code_audit = build_code_audit(lib, train_pairs, val_pairs)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "pairing_audit": pairing_audit,
        "code_audit": code_audit,
        "verdict": classify_issue(pairing_audit, code_audit),
    }


def main() -> None:
    args = parse_args()
    report = build_report(args.data_root.resolve())
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
