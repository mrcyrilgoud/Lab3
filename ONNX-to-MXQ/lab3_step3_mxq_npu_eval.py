#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MXQ_NAME = "best.mxq"
DEFAULT_ONNX_NAME = "best.onnx"
DEFAULT_INPUT_HEIGHT = 256
DEFAULT_INPUT_WIDTH = 256
DEFAULT_LR_DIR = "Data/LR_val"
DEFAULT_HR_DIR = "Data/HR_val"
WARMUP_RUNS = 3
SCRIPT_DIR = Path(__file__).resolve().parent

BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def collect_search_roots(*bases: Path) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        for candidate in [base, *base.parents]:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            roots.append(resolved)
    return roots


SEARCH_ROOTS = collect_search_roots(Path.cwd(), SCRIPT_DIR)


def resolve_existing_path(path_value: str | Path, expect_dir: bool) -> Path | None:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        candidates = [candidate.resolve()]
    else:
        candidates = [(root / candidate).resolve() for root in SEARCH_ROOTS]

    for resolved in candidates:
        if expect_dir and resolved.is_dir():
            return resolved
        if not expect_dir and resolved.is_file():
            return resolved
    return None


def resolve_default_mxq() -> Path | None:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in SEARCH_ROOTS:
        for candidate in [(root / DEFAULT_MXQ_NAME), (root / "exports" / DEFAULT_MXQ_NAME)]:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_model_path(cli_value: str | None) -> Path:
    if cli_value:
        resolved = resolve_existing_path(cli_value, expect_dir=False)
        if resolved is None:
            raise FileNotFoundError(f"MXQ model not found: {cli_value}")
        return resolved

    resolved = resolve_default_mxq()
    if resolved is None:
        raise FileNotFoundError(
            "Could not find MXQ model. Pass --mxq-model or place best.mxq in the working directory or exports/."
        )
    return resolved


def sibling_onnx_candidates(mxq_model: Path) -> list[Path]:
    return [
        mxq_model.with_suffix(".onnx"),
        mxq_model.with_name(DEFAULT_ONNX_NAME),
    ]


def infer_onnx_hw_from_sibling(mxq_model: Path, fallback: tuple[int, int]) -> tuple[int, int]:
    onnx_path = None
    for candidate in sibling_onnx_candidates(mxq_model):
        if candidate.is_file():
            onnx_path = candidate
            break
    if onnx_path is None:
        return fallback

    try:
        import onnx  # type: ignore

        model = onnx.load(str(onnx_path))
        dims = model.graph.input[0].type.tensor_type.shape.dim
        if len(dims) >= 4:
            height = int(dims[2].dim_value or 0)
            width = int(dims[3].dim_value or 0)
            if height > 0 and width > 0:
                return (height, width)
    except Exception as exc:
        print(f"Warning: failed to infer input shape from {onnx_path}: {exc}")

    return fallback


def collect_image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    images = sorted(
        file_path for file_path in path.iterdir() if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No images found under: {path}")
    return images


def collect_paired_samples(lr_dir: Path, hr_dir: Path, limit: int) -> list[tuple[Path, Path]]:
    lr_map = {path.stem: path for path in collect_image_files(lr_dir)}
    hr_map = {path.stem: path for path in collect_image_files(hr_dir)}
    matched_stems = sorted(lr_map.keys() & hr_map.keys())
    if limit > 0:
        matched_stems = matched_stems[:limit]

    pairs = [(lr_map[stem], hr_map[stem]) for stem in matched_stems]
    if not pairs:
        raise RuntimeError(f"No matched LR/HR pairs found.\n  LR: {lr_dir}\n  HR: {hr_dir}")
    return pairs


def load_lr_tensor(image_path: Path, input_hw: tuple[int, int]) -> np.ndarray:
    height, width = input_hw
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr.transpose(2, 0, 1), axis=0).astype(np.float32)


def load_hr_reference(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def unwrap_output_tensor(output: object) -> np.ndarray:
    candidate = output
    if isinstance(candidate, dict):
        if not candidate:
            raise RuntimeError("Model output dict is empty.")
        candidate = next(iter(candidate.values()))

    if isinstance(candidate, (list, tuple)):
        if not candidate:
            raise RuntimeError("Model output list/tuple is empty.")
        candidate = candidate[0]

    tensor = np.asarray(candidate, dtype=np.float32)
    if tensor.size == 0:
        raise RuntimeError("Model output tensor is empty.")
    return tensor


def output_to_hwc_image(output: object) -> np.ndarray:
    tensor = unwrap_output_tensor(output)

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise RuntimeError(f"Expected batch size 1 output, got shape {tensor.shape}")
        tensor = tensor[0]

    if tensor.ndim != 3:
        raise RuntimeError(f"Unexpected SR output shape: {tensor.shape}")

    if tensor.shape[0] in (1, 3):
        image = np.transpose(tensor, (1, 2, 0))
    elif tensor.shape[-1] in (1, 3):
        image = tensor
    else:
        raise RuntimeError(f"Could not interpret SR output layout: {tensor.shape}")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape:
        raise RuntimeError(f"Prediction/target shape mismatch for PSNR: pred={pred.shape}, target={target.shape}")

    mse = float(np.mean((pred - target) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def load_maccel():
    try:
        import maccel  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: 'maccel'. Run this script in the Mobilint NPU environment where maccel is installed."
        ) from exc
    return maccel


def run_infer(model: Any, input_tensor: np.ndarray) -> object:
    try:
        return model.infer([input_tensor])
    except TypeError:
        return model.infer(input_tensor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab 3 MXQ validation on the Mobilint NPU.")
    parser.add_argument("--mxq-model", type=str, default=None, help="Path to the MXQ model. Defaults to best.mxq in the working directory or exports/.")
    parser.add_argument("--lr-dir", type=str, default=DEFAULT_LR_DIR, help=f"LR validation directory (default: {DEFAULT_LR_DIR}).")
    parser.add_argument("--hr-dir", type=str, default=DEFAULT_HR_DIR, help=f"HR validation directory (default: {DEFAULT_HR_DIR}).")
    parser.add_argument("--device-index", type=int, default=0, help="NPU device index (default: 0).")
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_HEIGHT, help=f"Fallback model input height (default: {DEFAULT_INPUT_HEIGHT}).")
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_WIDTH, help=f"Fallback model input width (default: {DEFAULT_INPUT_WIDTH}).")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of matched validation pairs to evaluate (0 = all).")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional JSON path to write the evaluation summary.")
    return parser


def emit_summary(summary: dict[str, Any], summary_path: Path | None) -> None:
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if summary_path is not None:
        summary_path = summary_path.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.input_height <= 0 or args.input_width <= 0:
        raise ValueError("--input-height and --input-width must be > 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    mxq_model = resolve_model_path(args.mxq_model)
    input_hw = infer_onnx_hw_from_sibling(mxq_model, (args.input_height, args.input_width))
    if input_hw != (DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH):
        raise ValueError(
            f"Lab 3 requires a 256x256 model input contract, but resolved input shape was {input_hw[0]}x{input_hw[1]}."
        )

    lr_dir = resolve_existing_path(args.lr_dir, expect_dir=True)
    hr_dir = resolve_existing_path(args.hr_dir, expect_dir=True)
    if lr_dir is None:
        raise FileNotFoundError(f"LR validation directory not found: {args.lr_dir}")
    if hr_dir is None:
        raise FileNotFoundError(f"HR validation directory not found: {args.hr_dir}")

    pairs = collect_paired_samples(lr_dir, hr_dir, limit=args.limit)

    maccel = load_maccel()
    device = maccel.Accelerator(args.device_index)
    model = maccel.Model(str(mxq_model))
    model.launch(device)

    try:
        warmup_tensor = load_lr_tensor(pairs[0][0], input_hw)
        for _ in range(WARMUP_RUNS):
            run_infer(model, warmup_tensor)

        latencies_ms: list[float] = []
        psnr_values: list[float] = []
        for lr_path, hr_path in pairs:
            lr_tensor = load_lr_tensor(lr_path, input_hw)
            hr_image = load_hr_reference(hr_path)

            start = time.perf_counter()
            output = run_infer(model, lr_tensor)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

            pred_image = output_to_hwc_image(output)
            psnr_values.append(psnr(pred_image, hr_image))
    finally:
        model.dispose()

    mean_latency = statistics.mean(latencies_ms)
    mean_psnr = statistics.mean(psnr_values)
    throughput = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    summary = {
        "status": "completed",
        "mxq_model": str(mxq_model),
        "lr_dir": str(lr_dir),
        "hr_dir": str(hr_dir),
        "pair_count": len(pairs),
        "resolved_input_hw": [input_hw[0], input_hw[1]],
        "device_index": args.device_index,
        "warmup_runs": WARMUP_RUNS,
        "mean_psnr_db": mean_psnr,
        "mean_latency_ms": mean_latency,
        "throughput_fps": throughput,
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
        "summary_path": str(args.summary_path.expanduser().resolve()) if args.summary_path is not None else None,
    }
    emit_summary(summary, args.summary_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, ModuleNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
