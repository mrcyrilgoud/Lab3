from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data"
RUNS_ROOT = PROJECT_ROOT / "runs"
REPORTS_ROOT = RUNS_ROOT / "autopilot_reports"


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = max(float(np.mean((np.clip(a, 0.0, 1.0) - np.clip(b, 0.0, 1.0)) ** 2)), 1e-12)
    return float(-10.0 * math.log10(mse))


def _paired_val_stems() -> list[str]:
    hr_map = {path.stem: path for path in (DATA_ROOT / "HR_val").glob("*.png")}
    lr_map = {path.stem: path for path in (DATA_ROOT / "LR_val").glob("*.png")}
    return sorted(set(hr_map) & set(lr_map))


def _paired_maps() -> tuple[dict[str, Path], dict[str, Path]]:
    hr_map = {path.stem: path for path in (DATA_ROOT / "HR_val").glob("*.png")}
    lr_map = {path.stem: path for path in (DATA_ROOT / "LR_val").glob("*.png")}
    return lr_map, hr_map


def _mean_input_psnr(stems: list[str]) -> float:
    lr_map, hr_map = _paired_maps()
    values = [_psnr(_load_rgb(lr_map[stem]), _load_rgb(hr_map[stem])) for stem in stems]
    return float(sum(values) / max(1, len(values)))


def _image_sizes(directory: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(directory.glob("*.png")):
        with Image.open(path) as image:
            key = f"{image.size[0]}x{image.size[1]}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _best_contiguous_window(target: float, width: int) -> dict[str, Any]:
    stems = _paired_val_stems()
    lr_map, hr_map = _paired_maps()
    per_stem = {stem: _psnr(_load_rgb(lr_map[stem]), _load_rgb(hr_map[stem])) for stem in stems}
    best_gap = float("inf")
    best_payload: dict[str, Any] | None = None
    for start in range(0, len(stems) - width + 1):
        window = stems[start : start + width]
        mean_psnr = float(sum(per_stem[stem] for stem in window) / width)
        gap = abs(mean_psnr - target)
        if gap < best_gap:
            best_gap = gap
            best_payload = {
                "start_index": start,
                "window_size": width,
                "mean_input_psnr": mean_psnr,
                "abs_gap_to_target": gap,
                "stems": window,
            }
    assert best_payload is not None
    return best_payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_span_history() -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for summary_path in sorted(RUNS_ROOT.glob("**/span_*/summary.json")):
        payload = _load_json(summary_path)
        history.append(
            {
                "summary_path": str(summary_path),
                "run_root": str(summary_path.parent),
                "candidate_id": payload.get("candidate", {}).get("candidate_id"),
                "config": payload.get("config", {}),
                "evaluation": payload.get("evaluation", {}),
                "training": payload.get("training", {}),
            }
        )
    return history


def _load_baseline_refs() -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    ledger_path = REPORTS_ROOT / "ledger.jsonl"
    if not ledger_path.exists():
        return refs
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("candidate", {}).get("candidate_id") != "wide_residual_nobn_v1":
            continue
        sig = payload.get("comparison_signature", {})
        key = f"train{sig.get('train_pairs')}_val{sig.get('val_pairs')}"
        candidate = {
            "run_root": payload.get("run_root"),
            "train_pairs": sig.get("train_pairs"),
            "val_pairs": sig.get("val_pairs"),
            "num_epochs": sig.get("num_epochs"),
            "batch_size": sig.get("batch_size"),
            "val_psnr": payload.get("validation_psnr"),
            "input_psnr": payload.get("input_psnr"),
            "delta_psnr": payload.get("delta_psnr"),
        }
        prior = refs.get(key)
        if prior is None or (candidate.get("val_psnr") or float("-inf")) > (prior.get("val_psnr") or float("-inf")):
            refs[key] = candidate
    return refs


def build_report() -> dict[str, Any]:
    stems_lex = _paired_val_stems()
    stems_num = sorted(stems_lex, key=int)
    first_16_lex = stems_lex[:16]
    first_16_num = stems_num[:16]

    report: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "validation_inventory": {
            "pair_count": len(stems_lex),
            "hr_sizes": _image_sizes(DATA_ROOT / "HR_val"),
            "lr_sizes": _image_sizes(DATA_ROOT / "LR_val"),
            "first_16_lex_stems": first_16_lex,
            "first_16_num_stems": first_16_num,
            "input_psnr_first_16_lex": _mean_input_psnr(first_16_lex),
            "input_psnr_first_16_numeric": _mean_input_psnr(first_16_num),
            "input_psnr_all_100_lex": _mean_input_psnr(stems_lex),
        },
        "baseline_references": _load_baseline_refs(),
        "span_history": _load_span_history(),
    }

    for item in report["span_history"]:
        val_cap = item.get("config", {}).get("val_pair_cap")
        if isinstance(val_cap, int) and val_cap > 0:
            expected = _mean_input_psnr(stems_lex[:val_cap])
            logged = item.get("evaluation", {}).get("input_psnr")
            item["expected_input_psnr_from_current_lex_slice"] = expected
            item["input_psnr_gap_vs_current_lex_slice"] = None if logged is None else float(logged - expected)
            item["closest_contiguous_window"] = _best_contiguous_window(float(logged), val_cap) if logged is not None else None
        else:
            expected = _mean_input_psnr(stems_lex)
            logged = item.get("evaluation", {}).get("input_psnr")
            item["expected_input_psnr_from_current_lex_slice"] = expected
            item["input_psnr_gap_vs_current_lex_slice"] = None if logged is None else float(logged - expected)

    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
