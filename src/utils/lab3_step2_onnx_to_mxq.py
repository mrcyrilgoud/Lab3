from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def find_mxq_tool() -> str | None:
    for name in ["mxq_compile", "qubee", "qb"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 3 ONNX to MXQ handoff helper.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument(
        "--command-template",
        type=str,
        default="",
        help="Optional shell template with placeholders: {onnx}, {calibration}, {output}",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.onnx = args.onnx.resolve()
    args.calibration_dir = args.calibration_dir.resolve()
    args.output = args.output.resolve()

    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")
    if not args.calibration_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {args.calibration_dir}")

    tool = find_mxq_tool()
    payload = {
        "onnx": str(args.onnx),
        "calibration_dir": str(args.calibration_dir),
        "output": str(args.output),
        "mxq_tool_detected": tool,
        "status": "handoff_only",
    }

    if not tool and not args.command_template:
        payload["message"] = (
            "No MXQ compiler was detected in PATH. Provide --command-template if you want this helper "
            "to execute the vendor compiler command in your environment."
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command_template:
        command = args.command_template.format(
            onnx=shlex.quote(str(args.onnx)),
            calibration=shlex.quote(str(args.calibration_dir)),
            output=shlex.quote(str(args.output)),
        )
    else:
        command_parts = [
            shlex.quote(tool),
            shlex.quote(str(args.onnx)),
            shlex.quote(str(args.calibration_dir)),
            shlex.quote(str(args.output)),
        ]
        command_parts.extend(shlex.quote(arg) for arg in args.extra_arg)
        command = " ".join(command_parts)

    payload["status"] = "dry_run" if args.dry_run else "running"
    payload["command"] = command
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    completed = subprocess.run(command, shell=True, capture_output=True, text=True)
    payload["returncode"] = completed.returncode
    payload["stdout"] = completed.stdout
    payload["stderr"] = completed.stderr
    payload["status"] = "completed" if completed.returncode == 0 else "failed"
    payload["output_exists"] = args.output.exists()
    print(json.dumps(payload, indent=2))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
