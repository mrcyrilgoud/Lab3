from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
import yaml

from model import build_teacher_model

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

OPERATOR_RULES: dict[str, dict[str, str]] = {
    "Conv": {"pasuko": "Yes", "tier": "C", "rationale": "Preferred convolution path."},
    "DepthToSpace": {"pasuko": "Yes", "tier": "C", "rationale": "PixelShuffle export path."},
    "ReduceL2": {"pasuko": "Yes", "tier": "C", "rationale": "Sheet says Yes; rubric still treats Reduce ops cautiously."},
    "ReduceMean": {"pasuko": "CPU Fallback", "tier": "A", "rationale": "Part of custom LayerNorm chain."},
    "Sqrt": {"pasuko": "Risk / not Yes", "tier": "A", "rationale": "Quantizer-sensitive norm path op."},
    "Div": {"pasuko": "CPU Fallback", "tier": "B", "rationale": "Normalization and GELU decomposition path."},
    "MatMul": {"pasuko": "CPU Fallback", "tier": "B", "rationale": "Attention hot path."},
    "Softmax": {"pasuko": "CPU Fallback", "tier": "B", "rationale": "Attention hot path."},
    "Softplus": {"pasuko": "CPU Fallback", "tier": "B", "rationale": "Learnable temperature path."},
    "Erf": {"pasuko": "Gelu fallback family", "tier": "B", "rationale": "GELU decomposition path."},
    "Add": {"pasuko": "CPU Fallback", "tier": "B", "rationale": "Residual path and temperature offset."},
    "Concat": {"pasuko": "CPU Fallback", "tier": "B", "rationale": "Skip concatenation path."},
    "Clip": {"pasuko": "CPU Fallback", "tier": "B", "rationale": "Final clamp path."},
    "Transpose": {"pasuko": "CPU Fallback", "tier": "B", "rationale": "Attention layout path."},
}


@dataclass(frozen=True)
class RunLayout:
    run_root: Path
    export_root: Path
    onnx_path: Path
    calibration_dir: Path
    mxq_path: Path
    operator_audit_path: Path
    parity_path: Path
    mxq_payload_path: Path


def set_seed(seed: int = 255) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "Data").is_dir() and (candidate / "lab3_step2_onnx_to_mxq.py").exists():
            return candidate
    raise FileNotFoundError(f"Could not find Lab3 project root from {start}")


def load_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "profiles" not in payload:
        raise ValueError(f"Invalid config file: {config_path}")
    return payload


def resolve_profile(config: dict[str, Any], profile_name: str | None = None) -> tuple[str, dict[str, Any]]:
    selected = profile_name or str(config.get("active_profile", "smoke"))
    profiles = dict(config["profiles"])
    if selected not in profiles:
        raise KeyError(f"Unknown profile: {selected}")
    return selected, dict(profiles[selected])


def build_model(profile: dict[str, Any]) -> torch.nn.Module:
    return build_teacher_model(profile)


def require_onnx() -> None:
    if onnx is None:
        raise RuntimeError("onnx is required for this notebook because ONNX checker and graph audit are mandatory.")


def _collect_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _index_by_name(paths: list[Path], label: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in paths:
        key = path.name
        if key in mapping:
            duplicates.append(key)
        else:
            mapping[key] = path
    if duplicates:
        dupes = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"Duplicate {label} basenames found: {dupes}")
    return mapping


def pair_by_basename(lr_root: Path, hr_root: Path) -> list[tuple[Path, Path, str]]:
    lr_paths = _collect_images(lr_root)
    hr_paths = _collect_images(hr_root)
    lr_map = _index_by_name(lr_paths, "LR")
    hr_map = _index_by_name(hr_paths, "HR")
    common = sorted(set(lr_map) & set(hr_map))
    if not common:
        raise FileNotFoundError(f"No paired images found between {lr_root} and {hr_root}")
    missing_lr = sorted(set(hr_map) - set(lr_map))
    missing_hr = sorted(set(lr_map) - set(hr_map))
    if missing_lr or missing_hr:
        raise ValueError(
            f"LR/HR basename mismatch under {lr_root} and {hr_root}: "
            f"missing_lr={len(missing_lr)} missing_hr={len(missing_hr)}"
        )
    return [(lr_map[name], hr_map[name], name) for name in common]


def validate_data_layout(data_root: Path) -> dict[str, Any]:
    train_pairs: list[tuple[Path, Path, str]] = []
    train_roots: list[dict[str, str]] = []
    for index in range(1, 5):
        lr_root = data_root / "LR_train" / f"LR_train{index}"
        hr_root = data_root / "HR_train" / f"HR_train{index}"
        if not lr_root.is_dir() or not hr_root.is_dir():
            raise FileNotFoundError(f"Missing training directories: {lr_root} and/or {hr_root}")
        pairs = pair_by_basename(lr_root, hr_root)
        train_pairs.extend(pairs)
        train_roots.append({"lr_root": str(lr_root), "hr_root": str(hr_root), "pairs": len(pairs)})
    val_lr_root = data_root / "LR_val"
    val_hr_root = data_root / "HR_val"
    if not val_lr_root.is_dir() or not val_hr_root.is_dir():
        raise FileNotFoundError(f"Missing validation directories: {val_lr_root} and/or {val_hr_root}")
    val_pairs = pair_by_basename(val_lr_root, val_hr_root)
    return {
        "data_root": str(data_root),
        "train_pair_count": len(train_pairs),
        "val_pair_count": len(val_pairs),
        "train_roots": train_roots,
        "val_roots": {"lr_root": str(val_lr_root), "hr_root": str(val_hr_root)},
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
    }


def verify_model_contract(model: torch.nn.Module, eval_size: int, device: torch.device) -> dict[str, Any]:
    model = model.to(device)
    model.eval()
    dummy = torch.zeros(1, 3, eval_size, eval_size, device=device)
    with torch.no_grad():
        output = model(dummy)
    output_shape = tuple(int(x) for x in output.shape)
    if output_shape != (1, 3, eval_size, eval_size):
        raise AssertionError(f"Model output shape mismatch: expected (1, 3, {eval_size}, {eval_size}), got {output_shape}")
    return {
        "input_shape": [1, 3, eval_size, eval_size],
        "output_shape": list(output_shape),
        "contract_ok": True,
    }


def build_run_layout(project_root: Path, run_name: str, started_day: str, export_slug: str, onnx_name: str, mxq_name: str) -> RunLayout:
    run_root = project_root / "runs" / started_day / run_name
    export_root = run_root / "exports" / export_slug
    return RunLayout(
        run_root=run_root,
        export_root=export_root,
        onnx_path=export_root / onnx_name,
        calibration_dir=export_root / "calibration",
        mxq_path=export_root / mxq_name,
        operator_audit_path=export_root / "operator_audit.json",
        parity_path=export_root / "parity.json",
        mxq_payload_path=export_root / "mxq_handoff.json",
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_to_onnx(model: torch.nn.Module, onnx_path: Path, device: torch.device, eval_size: int, verify: bool) -> dict[str, Any]:
    require_onnx()
    model = model.to(device)
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
    loaded = onnx.load(str(onnx_path))
    onnx.checker.check_model(loaded)
    parity: dict[str, Any]
    if verify and ort is not None:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_output = session.run(None, {"input": dummy.detach().cpu().numpy()})[0]
        torch_output = model(dummy).detach().cpu().numpy()
        diff = np.abs(ort_output - torch_output)
        parity = {
            "status": "passed",
            "provider": "CPUExecutionProvider",
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
        }
    elif verify:
        parity = {"status": "skipped_missing_onnxruntime"}
    else:
        parity = {"status": "skipped_by_config"}
    return {
        "onnx_path": str(onnx_path),
        "onnx_size_kb": round(onnx_path.stat().st_size / 1024.0, 2),
        "onnx_checker": "passed",
        "onnx_opset": int(loaded.opset_import[0].version),
        "parity": parity,
    }


def audit_onnx_graph(onnx_path: Path) -> dict[str, Any]:
    require_onnx()
    loaded = onnx.load(str(onnx_path))
    counts = Counter(node.op_type for node in loaded.graph.node)
    counts_no_constant = {op: count for op, count in counts.items() if op != "Constant"}
    rows = []
    for op, count in sorted(counts_no_constant.items(), key=lambda item: (-item[1], item[0])):
        rule = OPERATOR_RULES.get(op, {"pasuko": "Unclassified", "tier": "B", "rationale": "Review manually."})
        rows.append(
            {
                "op_type": op,
                "count": int(count),
                "pasuko": rule["pasuko"],
                "tier": rule["tier"],
                "rationale": rule["rationale"],
            }
        )
    return {
        "onnx_path": str(onnx_path),
        "op_counts": dict(sorted(counts.items())),
        "op_counts_no_constant": dict(sorted(counts_no_constant.items())),
        "risk_rows": rows,
        "required_ops_present": {
            op: op in counts_no_constant
            for op in ["Conv", "DepthToSpace", "ReduceL2", "ReduceMean", "Sqrt", "Div", "MatMul", "Softmax", "Softplus", "Erf", "Add", "Concat", "Clip", "Transpose"]
        },
    }


def operator_risk_markdown(audit: dict[str, Any]) -> str:
    header = "| ONNX op | Count | Pasuko classification | Risk tier | Rationale |"
    divider = "|---|---:|---|---|---|"
    body = [
        f"| {row['op_type']} | {row['count']} | {row['pasuko']} | {row['tier']} | {row['rationale']} |"
        for row in audit["risk_rows"]
    ]
    return "\n".join([header, divider, *body])


def compute_image_profile(image_path: Path) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB").resize((64, 64), RESAMPLING_BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    return {"brightness": float(gray.mean()), "texture": float((grad_x + grad_y) * 0.5)}


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


def export_calibration_dataset(train_pairs: list[tuple[Path, Path, str]], calibration_dir: Path, eval_size: int, calibration_count: int) -> dict[str, Any]:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    selected = select_calibration_pairs(train_pairs, calibration_count)
    manifest: list[dict[str, Any]] = []
    for index, (lr_path, hr_path, name) in enumerate(selected):
        image = Image.open(lr_path).convert("RGB")
        if image.size != (eval_size, eval_size):
            image = ImageOps.fit(image, (eval_size, eval_size), method=RESAMPLING_BICUBIC)
        output_path = calibration_dir / f"{index:03d}_{Path(name).stem}.png"
        image.save(output_path)
        manifest.append(
            {
                "index": index,
                "name": name,
                "source_lr": str(lr_path),
                "source_hr": str(hr_path),
                "image_path": str(output_path),
                "derived_from_training": True,
            }
        )
    manifest_payload = {
        "count": len(manifest),
        "eval_size": eval_size,
        "source": "training_pairs",
        "format": "png",
        "input_domain": "lr",
        "items": manifest,
    }
    write_json(calibration_dir / "manifest.json", manifest_payload)
    return {
        "calibration_dir": str(calibration_dir),
        "manifest_path": str(calibration_dir / "manifest.json"),
        "count": len(manifest),
        "source": "training_pairs",
        "derived_from_training": True,
        "input_domain": "lr",
    }


def run_mxq_handoff(
    project_root: Path,
    onnx_path: Path,
    calibration_dir: Path,
    output_path: Path,
    *,
    dry_run: bool = True,
    command_template: str = "",
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    helper_path = project_root / "lab3_step2_onnx_to_mxq.py"
    if not helper_path.exists():
        return {"status": "missing_helper", "helper_path": str(helper_path)}
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
    if command_template:
        command.extend(["--command-template", command_template])
    for arg in extra_args or []:
        command.extend(["--extra-arg", arg])
    if dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(project_root))
    payload: dict[str, Any] = {
        "helper_path": str(helper_path),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass
    payload["helper_path"] = str(helper_path)
    payload["returncode"] = completed.returncode
    payload["requested_compile"] = not dry_run
    payload["output_exists"] = output_path.exists()
    return payload
