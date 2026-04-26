#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert or hand off HiNetLite v5-clean ONNX to MXQ.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument(
        "--command-template",
        default="",
        help="Optional command with {onnx}, {calibration}, and {output} placeholders.",
    )
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_repo_helper(start: Path) -> Path | None:
    for root in [start, *start.parents]:
        helper = root / "src" / "utils" / "lab3_step2_onnx_to_mxq.py"
        if helper.exists():
            return helper
    return None


def find_mxq_tool() -> str | None:
    for name in ["mxq_compile", "qubee", "qb"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def fallback_payload(args: argparse.Namespace) -> dict[str, object]:
    tool = find_mxq_tool()
    payload: dict[str, object] = {
        "onnx": str(args.onnx),
        "calibration_dir": str(args.calibration_dir),
        "output": str(args.output),
        "mxq_tool_detected": tool,
        "status": "handoff_only",
    }
    if not tool and not args.command_template:
        payload["message"] = "No MXQ compiler detected. Provide --command-template in the Mobilint environment."
        return payload

    if args.command_template:
        command = args.command_template.format(
            onnx=shlex.quote(str(args.onnx)),
            calibration=shlex.quote(str(args.calibration_dir)),
            output=shlex.quote(str(args.output)),
        )
    else:
        parts = [shlex.quote(str(tool)), shlex.quote(str(args.onnx)), shlex.quote(str(args.calibration_dir)), shlex.quote(str(args.output))]
        parts.extend(shlex.quote(item) for item in args.extra_arg)
        command = " ".join(parts)

    payload["command"] = command
    payload["status"] = "dry_run" if args.dry_run else "running"
    if args.dry_run:
        return payload

    completed = subprocess.run(command, shell=True, capture_output=True, text=True)
    payload.update(
        {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "status": "completed" if completed.returncode == 0 else "failed",
            "output_exists": args.output.exists(),
        }
    )
    return payload


def run_helper(args: argparse.Namespace, helper: Path) -> dict[str, object]:
    command = [
        sys.executable,
        str(helper),
        "--onnx",
        str(args.onnx),
        "--calibration-dir",
        str(args.calibration_dir),
        "--output",
        str(args.output),
    ]
    if args.command_template:
        command.extend(["--command-template", args.command_template])
    for extra_arg in args.extra_arg:
        command.extend(["--extra-arg", extra_arg])
    if args.dry_run:
        command.append("--dry-run")

    completed = subprocess.run(command, capture_output=True, text=True)
    try:
        payload = json.loads(completed.stdout) if completed.stdout.strip() else {}
    except json.JSONDecodeError:
        payload = {}
    payload.update(
        {
            "helper_path": str(helper),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "output_exists": args.output.exists(),
        }
    )
    if "status" not in payload:
        payload["status"] = "completed" if completed.returncode == 0 else "failed"
    return payload


def main() -> None:
    args = parse_args()
    args.onnx = args.onnx.resolve()
    args.calibration_dir = args.calibration_dir.resolve()
    args.output = args.output.resolve()
    if args.log_path is None:
        args.log_path = args.output.with_suffix(".conversion_log.json")
    else:
        args.log_path = args.log_path.resolve()

    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")
    if not args.calibration_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {args.calibration_dir}")

    helper = find_repo_helper(Path.cwd())
    payload = run_helper(args, helper) if helper is not None else fallback_payload(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text(text + "\n", encoding="utf-8")
    if payload.get("returncode", 0) not in {0, None}:
        raise SystemExit(int(payload["returncode"]))


if __name__ == "__main__":
    main()
