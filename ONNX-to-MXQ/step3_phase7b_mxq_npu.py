#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MXQ_NAME = "best.mxq"
DEFAULT_ONNX_NAME = "best.onnx"
DEFAULT_INPUT_HEIGHT = 256
DEFAULT_INPUT_WIDTH = 256
DEFAULT_LR_DIR = "DataVal/LR_val"
DEFAULT_HR_DIR = "DataVal/HR_val"
WARMUP_RUNS = 3

BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
SCRIPT_DIR = Path(__file__).resolve().parent


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
    candidates = [candidate.resolve()] if candidate.is_absolute() else [(root / candidate).resolve() for root in SEARCH_ROOTS]

    for resolved in candidates:
        if expect_dir and resolved.is_dir():
            return resolved
        if not expect_dir and resolved.is_file():
            return resolved
    return None


def resolve_model_path(cli_value: str | None) -> Path:
    if cli_value:
        resolved = resolve_existing_path(cli_value, expect_dir=False)
        if resolved is None:
            raise FileNotFoundError(f"MXQ model not found: {cli_value}")
        return resolved

    matches: list[Path] = []
    seen: set[Path] = set()
    for root in SEARCH_ROOTS:
        candidate = (root / DEFAULT_MXQ_NAME).resolve()
        if candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            matches.append(candidate)

    if not matches:
        raise FileNotFoundError(
            f"Could not find MXQ model. Pass --mxq-model or place {DEFAULT_MXQ_NAME} in the working directory."
        )
    if len(matches) > 1:
        print(f"Warning: multiple MXQ files found, using {matches[0]}")
    return matches[0]


def infer_onnx_hw_from_sibling(mxq_model: Path, fallback: tuple[int, int]) -> tuple[int, int]:
    onnx_path = mxq_model.with_name(DEFAULT_ONNX_NAME)
    if not onnx_path.is_file():
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
            "Missing dependency: 'maccel'. Run this script in the NPU environment where maccel is installed."
        ) from exc
    return maccel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 7B MXQ validation on NPU.")
    parser.add_argument("--mxq-model", type=str, help="Path to the MXQ model. Defaults to ./best.mxq.")
    parser.add_argument("--lr-dir", type=str, default=DEFAULT_LR_DIR, help=f"LR validation directory (default: {DEFAULT_LR_DIR}).")
    parser.add_argument("--hr-dir", type=str, default=DEFAULT_HR_DIR, help=f"HR validation directory (default: {DEFAULT_HR_DIR}).")
    parser.add_argument("--device-index", type=int, default=0, help="NPU device index (default: 0).")
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_HEIGHT, help=f"Fallback model input height (default: {DEFAULT_INPUT_HEIGHT}).")
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_WIDTH, help=f"Fallback model input width (default: {DEFAULT_INPUT_WIDTH}).")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of matched validation pairs to evaluate (0 = all).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.input_height <= 0 or args.input_width <= 0:
        raise ValueError("--input-height and --input-width must be > 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    mxq_model = resolve_model_path(args.mxq_model)
    input_hw = infer_onnx_hw_from_sibling(mxq_model, (args.input_height, args.input_width))
    lr_dir = resolve_existing_path(args.lr_dir, expect_dir=True)
    hr_dir = resolve_existing_path(args.hr_dir, expect_dir=True)
    if lr_dir is None:
        raise FileNotFoundError(f"LR validation directory not found: {args.lr_dir}")
    if hr_dir is None:
        raise FileNotFoundError(f"HR validation directory not found: {args.hr_dir}")

    pairs = collect_paired_samples(lr_dir, hr_dir, limit=args.limit)

    print(f"Using MXQ: {mxq_model}")
    print(f"Input HW:  {input_hw[0]}x{input_hw[1]}")
    print(f"Pairs:     {len(pairs)}")

    maccel = load_maccel()
    device = maccel.Accelerator(args.device_index)
    model = maccel.Model(str(mxq_model))
    model.launch(device)

    try:
        warmup_tensor = load_lr_tensor(pairs[0][0], input_hw)
        for _ in range(WARMUP_RUNS):
            model.infer([warmup_tensor])

        latencies_ms: list[float] = []
        psnr_values: list[float] = []
        for lr_path, hr_path in pairs:
            lr_tensor = load_lr_tensor(lr_path, input_hw)
            hr_image = load_hr_reference(hr_path)

            start = time.perf_counter()
            output = model.infer([lr_tensor])
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

            pred_image = output_to_hwc_image(output)
            psnr_values.append(psnr(pred_image, hr_image))
    finally:
        model.dispose()

    mean_latency = statistics.mean(latencies_ms)
    mean_psnr = statistics.mean(psnr_values)
    throughput = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    print(f"Mean PSNR:    {mean_psnr:.3f} dB")
    print(f"Mean latency: {mean_latency:.2f} ms/image")
    print(f"Throughput:   {throughput:.2f} images/sec")


if __name__ == "__main__":
    main()
