#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ONNX_NAME = "lab3.onnx"
DEFAULT_CALIBRATION_DIRNAME = "calibration"
LEGACY_CALIBRATION_DIRNAME = "phase7b_calibration_png_lr_dataset"
DEFAULT_CALIB_IMAGE_STAGING_DIRNAME = ".lab3_calibration_images_auto"
CALIB_DATA_DIR_PREFIX = ".lab3_calibration_data_"
DEFAULT_QUANTIZATION_OUTPUT_INDEX = 0
DEFAULT_INPUT_HEIGHT = 256
DEFAULT_INPUT_WIDTH = 256
SCRIPT_DIR = Path(__file__).resolve().parent

BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a Lab 3 ONNX export into MXQ using training-derived calibration data."
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Path to the ONNX model. Defaults to best.onnx in the working directory or exports/.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Path to the training-derived calibration image directory. Defaults to calibration/ next to the ONNX export.",
    )
    parser.add_argument(
        "--output-mxq",
        type=Path,
        default=None,
        help="Output MXQ path. Defaults to the ONNX path with a .mxq suffix.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=DEFAULT_INPUT_HEIGHT,
        help=f"Fallback model input height when ONNX shape inference is unavailable (default: {DEFAULT_INPUT_HEIGHT}).",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=DEFAULT_INPUT_WIDTH,
        help=f"Fallback model input width when ONNX shape inference is unavailable (default: {DEFAULT_INPUT_WIDTH}).",
    )
    parser.add_argument(
        "--quantize-method",
        default="maxpercentile",
        help="Qubee quantization method (default: maxpercentile).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.999,
        help="Quantization percentile (default: 0.999).",
    )
    parser.add_argument(
        "--topk-ratio",
        type=float,
        default=0.01,
        help="Top-k ratio for maxpercentile quantization (default: 0.01).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated staged calibration images and intermediate calibration data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print a JSON summary without invoking Qubee.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON path to write the conversion summary.",
    )
    return parser.parse_args()


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


def resolve_default_onnx() -> Path | None:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in SEARCH_ROOTS:
        for candidate in [(root / DEFAULT_ONNX_NAME), (root / "exports" / DEFAULT_ONNX_NAME)]:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def calibration_candidates_for_onnx(onnx_model: Path) -> list[Path]:
    parent = onnx_model.parent
    candidates = [
        parent / DEFAULT_CALIBRATION_DIRNAME,
        parent / LEGACY_CALIBRATION_DIRNAME,
    ]
    if parent.name != "exports":
        candidates.extend(
            [
                parent / "exports" / DEFAULT_CALIBRATION_DIRNAME,
                parent / "exports" / LEGACY_CALIBRATION_DIRNAME,
            ]
        )
    return [candidate.resolve() for candidate in candidates]


def resolve_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    onnx_model = (
        resolve_existing_path(args.onnx_model, expect_dir=False)
        if args.onnx_model is not None
        else resolve_default_onnx()
    )
    if onnx_model is None:
        raise FileNotFoundError(
            "ONNX model not found. Pass --onnx-model or place best.onnx in the working directory or exports/."
        )

    if args.calibration_dir is not None:
        calibration_dir = resolve_existing_path(args.calibration_dir, expect_dir=True)
        if calibration_dir is None:
            raise FileNotFoundError(f"Calibration directory not found: {args.calibration_dir}")
    else:
        calibration_dir = None
        for candidate in calibration_candidates_for_onnx(onnx_model):
            if candidate.is_dir():
                calibration_dir = candidate
                break
        if calibration_dir is None:
            raise FileNotFoundError(
                "Calibration directory not found. Pass --calibration-dir or place calibration/ next to the ONNX export."
            )

    if args.output_mxq is not None:
        output_mxq = args.output_mxq.expanduser().resolve()
    else:
        output_mxq = onnx_model.with_suffix(".mxq")

    summary_path = args.summary_path.expanduser().resolve() if args.summary_path is not None else None
    return {
        "onnx_model": onnx_model,
        "calibration_dir": calibration_dir,
        "output_mxq": output_mxq,
        "summary_path": summary_path,
    }


def load_manifest(calibration_dir: Path) -> dict[str, Any] | None:
    manifest_path = calibration_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Calibration manifest must be a JSON object: {manifest_path}")
    return manifest


def collect_calibration_images(calibration_dir: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in calibration_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def summarize_manifest(manifest: dict[str, Any] | None, manifest_path: Path) -> dict[str, Any]:
    if manifest is None:
        return {"present": False}

    items = manifest.get("items")
    samples = manifest.get("samples")
    summary = {
        "present": True,
        "path": str(manifest_path),
        "count": None,
        "derived_from_training": manifest.get("derived_from_training"),
        "source": manifest.get("source"),
    }
    if isinstance(items, list):
        summary["count"] = len(items)
    elif isinstance(samples, list):
        summary["count"] = len(samples)
    elif isinstance(manifest.get("count"), int):
        summary["count"] = int(manifest["count"])

    if summary["derived_from_training"] is None:
        if isinstance(manifest.get("summary"), dict):
            summary["derived_from_training"] = manifest["summary"].get("derived_from_training")
        elif isinstance(manifest.get("source"), str):
            summary["derived_from_training"] = manifest["source"] == "training_pairs"
    return summary


def validate_inputs(paths: dict[str, Path | None]) -> dict[str, Any]:
    onnx_model = paths["onnx_model"]
    calibration_dir = paths["calibration_dir"]
    assert onnx_model is not None
    assert calibration_dir is not None

    if not onnx_model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model}")
    if not calibration_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

    manifest_path = calibration_dir / "manifest.json"
    manifest = load_manifest(calibration_dir)
    calibration_images = collect_calibration_images(calibration_dir)
    if not calibration_images:
        raise RuntimeError(f"No calibration images found under: {calibration_dir}")

    manifest_summary = summarize_manifest(manifest, manifest_path)
    return {
        "manifest": manifest,
        "manifest_summary": manifest_summary,
        "calibration_images": calibration_images,
        "calibration_count": len(calibration_images),
    }


def sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def calibration_data_path_for_model(onnx_model: Path, temp_root: Path) -> Path:
    try:
        model_key = str(onnx_model.relative_to(temp_root.parent))
    except ValueError:
        model_key = onnx_model.name
    return temp_root / f"{CALIB_DATA_DIR_PREFIX}{sanitize_id(model_key)}"


def infer_onnx_hw(onnx_path: Path, fallback: tuple[int, int]) -> tuple[int, int]:
    try:
        import onnx  # type: ignore
    except Exception:
        return fallback

    try:
        model = onnx.load(str(onnx_path))
        first_input = model.graph.input[0]
        dims = first_input.type.tensor_type.shape.dim
        if len(dims) >= 4:
            height = int(dims[2].dim_value or 0)
            width = int(dims[3].dim_value or 0)
            if height > 0 and width > 0:
                return (height, width)
    except Exception as exc:
        print(f"Warning: failed to infer input shape from {onnx_path.name}: {exc}")
    return fallback


def build_preprocess(height: int, width: int):
    def _preprocess(image_path: str):
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            if rgb.size != (width, height):
                rgb = rgb.resize((width, height), BILINEAR)
            return (np.asarray(rgb, dtype=np.float32) / 255.0).astype(np.float32)

    return _preprocess


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def stage_calibration_images(sample_paths: Sequence[Path], stage_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(sample_paths):
        suffix = src.suffix.lower() if src.suffix else ".png"
        link_or_copy(src, stage_dir / f"{idx:06d}{suffix}")


def quantization_mode_index(quantize_method: str) -> int:
    normalized = re.sub(r"[^a-z0-9]+", "", quantize_method.lower())
    mode_map = {
        "percentile": 0,
        "max": 1,
        "maxpercentile": 2,
        "fastpercentile": 3,
        "histogramkl": 4,
        "histogrammse": 5,
    }
    if normalized not in mode_map:
        valid = ", ".join(sorted(mode_map))
        raise ValueError(
            f"Unsupported --quantize-method '{quantize_method}'. Valid values map to: {valid}"
        )
    return mode_map[normalized]


def call_mxq_compile_compatible(
    mxq_compile,
    onnx_model: Path,
    calib_data_path: Path,
    output_mxq: Path,
    quantize_method: str,
    percentile: float,
    topk_ratio: float,
) -> None:
    base_kwargs = {
        "model": str(onnx_model),
        "calib_data_path": str(calib_data_path),
        "topk_ratio": topk_ratio,
        "save_path": str(output_mxq),
        "backend": "onnx",
    }

    new_style_kwargs = {
        **base_kwargs,
        "quantization_mode": quantization_mode_index(quantize_method),
        "quantization_output": DEFAULT_QUANTIZATION_OUTPUT_INDEX,
        "percentile": percentile,
    }
    old_style_kwargs = {
        **base_kwargs,
        "quantize_method": quantize_method,
        "is_quant_ch": True,
        "quantize_percentile": percentile,
        "quant_output": "layer",
    }

    attempts = [("new", new_style_kwargs), ("old", old_style_kwargs)]
    try:
        params = inspect.signature(mxq_compile).parameters
        if "quantization_mode" in params:
            attempts = [("new", new_style_kwargs), ("old", old_style_kwargs)]
        elif "quantize_method" in params:
            attempts = [("old", old_style_kwargs), ("new", new_style_kwargs)]
    except (TypeError, ValueError):
        pass

    last_error = None
    for _, kwargs in attempts:
        try:
            mxq_compile(**kwargs)
            return
        except (TypeError, ValueError) as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("mxq_compile failed without an explicit error.")


def detect_toolchain() -> dict[str, Any]:
    qubee_importable = False
    qubee_error = None
    try:
        import qubee  # type: ignore  # noqa: F401

        qubee_importable = True
    except Exception as exc:
        qubee_error = str(exc)

    cli_detected = None
    for name in ["mxq_compile", "qubee", "qb"]:
        path = shutil.which(name)
        if path:
            cli_detected = path
            break

    return {
        "qubee_importable": qubee_importable,
        "qubee_import_error": qubee_error,
        "mxq_cli_detected": cli_detected,
    }


def compile_model(
    onnx_model: Path,
    output_mxq: Path,
    calib_image_dir: Path,
    temp_root: Path,
    input_hw: tuple[int, int],
    quantize_method: str,
    percentile: float,
    topk_ratio: float,
) -> Path:
    try:
        from qubee import mxq_compile
        from qubee.calibration import make_calib_man
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: 'qubee'. Install it in this environment before running MXQ compile."
        ) from exc

    calib_data_path = calibration_data_path_for_model(onnx_model, temp_root)
    if calib_data_path.exists():
        shutil.rmtree(calib_data_path)

    height, width = input_hw
    preprocess_fn = build_preprocess(height, width)
    calib_data_name = calib_data_path.name

    make_calib_man(
        pre_ftn=preprocess_fn,
        data_dir=str(calib_image_dir),
        save_dir=str(temp_root),
        save_name=calib_data_name,
        max_size=len(os.listdir(calib_image_dir)),
    )

    output_mxq.parent.mkdir(parents=True, exist_ok=True)
    call_mxq_compile_compatible(
        mxq_compile=mxq_compile,
        onnx_model=onnx_model,
        calib_data_path=calib_data_path,
        output_mxq=output_mxq,
        quantize_method=quantize_method,
        percentile=percentile,
        topk_ratio=topk_ratio,
    )
    return calib_data_path


def emit_payload(payload: dict[str, Any], summary_path: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.input_height <= 0 or args.input_width <= 0:
        raise ValueError("--input-height and --input-width must be > 0")

    summary_path = args.summary_path.expanduser().resolve() if args.summary_path is not None else None
    payload: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "status": "starting",
        "dry_run": bool(args.dry_run),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "config": {
            "input_height": int(args.input_height),
            "input_width": int(args.input_width),
            "quantize_method": args.quantize_method,
            "percentile": float(args.percentile),
            "topk_ratio": float(args.topk_ratio),
            "keep_temp": bool(args.keep_temp),
        },
        "toolchain": detect_toolchain(),
    }

    temp_root = SCRIPT_DIR
    staged_calib_dir = temp_root / DEFAULT_CALIB_IMAGE_STAGING_DIRNAME
    generated_calib_paths: list[Path] = []
    paths: dict[str, Path | None] = {
        "onnx_model": None,
        "calibration_dir": None,
        "output_mxq": None,
        "summary_path": summary_path,
    }

    try:
        paths = resolve_paths(args)
        summary_path = paths["summary_path"]
        assert isinstance(summary_path, (Path, type(None)))
        payload.update(
            {
                "onnx_model": str(paths["onnx_model"]),
                "calibration_dir": str(paths["calibration_dir"]),
                "output_mxq": str(paths["output_mxq"]),
            }
        )
        validation = validate_inputs(paths)
        onnx_model = paths["onnx_model"]
        calibration_dir = paths["calibration_dir"]
        output_mxq = paths["output_mxq"]
        assert isinstance(onnx_model, Path)
        assert isinstance(calibration_dir, Path)
        assert isinstance(output_mxq, Path)

        input_hw = infer_onnx_hw(onnx_model, fallback=(args.input_height, args.input_width))
        if input_hw != (DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH):
            raise ValueError(
                f"Lab 3 requires a 256x256 model input contract, but resolved input shape was {input_hw[0]}x{input_hw[1]}."
            )

        payload.update(
            {
                "resolved_input_hw": [input_hw[0], input_hw[1]],
                "contract_ok": True,
                "calibration_count": validation["calibration_count"],
                "manifest": validation["manifest_summary"],
                "output_exists": output_mxq.exists(),
            }
        )

        if args.dry_run:
            payload["status"] = "dry_run"
            emit_payload(payload, summary_path)
            return 0

        stage_calibration_images(validation["calibration_images"], staged_calib_dir)
        calib_data_path = compile_model(
            onnx_model=onnx_model,
            output_mxq=output_mxq,
            calib_image_dir=staged_calib_dir,
            temp_root=temp_root,
            input_hw=input_hw,
            quantize_method=args.quantize_method,
            percentile=args.percentile,
            topk_ratio=args.topk_ratio,
        )
        generated_calib_paths.append(calib_data_path)

        payload.update(
            {
                "status": "completed",
                "generated_calibration_data_path": str(calib_data_path),
                "output_exists": output_mxq.exists(),
                "output_size_mb": round(output_mxq.stat().st_size / (1024.0 * 1024.0), 3)
                if output_mxq.exists()
                else None,
            }
        )
        emit_payload(payload, summary_path)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, ModuleNotFoundError) as exc:
        payload.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "output_exists": Path(str(paths["output_mxq"])).exists() if paths["output_mxq"] is not None else False,
            }
        )
        emit_payload(payload, summary_path)
        return 1
    finally:
        if not args.keep_temp:
            if staged_calib_dir.exists():
                shutil.rmtree(staged_calib_dir)
            for calib_data_path in generated_calib_paths:
                if calib_data_path.exists():
                    shutil.rmtree(calib_data_path)


if __name__ == "__main__":
    raise SystemExit(main())
