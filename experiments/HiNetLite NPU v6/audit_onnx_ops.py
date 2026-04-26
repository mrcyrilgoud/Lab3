#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SUSPICIOUS_OPS = {
    "Mul",
    "Add",
    "Resize",
    "Concat",
    "MatMul",
    "Softmax",
    "Sigmoid",
    "Tanh",
    "InstanceNormalization",
    "GroupNormalization",
    "LayerNormalization",
    "ReduceMean",
    "Div",
    "Sub",
    "Pow",
    "Sqrt",
    "Reshape",
    "Transpose",
    "Slice",
    "Gather",
    "Unsqueeze",
    "Squeeze",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ONNX operators for the HiNetLite v5-clean path.")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to the ONNX model.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for a JSON audit report.")
    return parser.parse_args()


def audit_onnx_ops(onnx_path: Path) -> dict[str, object]:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("onnx is required: pip install onnx") from exc

    onnx_path = onnx_path.resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    ops = [node.op_type for node in model.graph.node]
    counts = dict(sorted(Counter(ops).items()))
    suspicious = {op: counts[op] for op in sorted(SUSPICIOUS_OPS) if op in counts}
    return {
        "onnx_path": str(onnx_path),
        "onnx_checker": "passed",
        "total_node_count": len(ops),
        "unique_op_types": sorted(counts),
        "op_counts": counts,
        "suspicious_ops_found": suspicious,
        "mul_found": "Mul" in counts,
        "add_found": "Add" in counts,
    }


def main() -> None:
    args = parse_args()
    report = audit_onnx_ops(args.onnx)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
