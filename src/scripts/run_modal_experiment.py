#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from notebook_execution import NotebookExecutionFailure, build_notebook_env, execute_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NOTEBOOK = PROJECT_ROOT / "lab3_wide_residual_nobn_modal_app.ipynb"


def import_candidate_registry() -> tuple[Any, Any]:
    tools_dir = PROJECT_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    from lab3_pipeline_lib import get_candidate_spec, list_candidate_ids  # type: ignore

    return get_candidate_spec, list_candidate_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a real Lab 3 Modal experiment through the canonical notebook.")
    parser.add_argument("--family", required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--kernel-mix", required=True)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--data-slice", required=True)
    parser.add_argument("--budget-minutes", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--train-patch-size", type=int, default=224)
    parser.add_argument("--modal-gpu", default="L40S")
    parser.add_argument("--modal-data-volume", default="lab3-data")
    parser.add_argument("--modal-runs-volume", default="lab3-runs")
    parser.add_argument("--poll-interval-minutes", type=int, default=0)
    parser.add_argument("--prior-best-val-psnr", type=float, default=None)
    parser.add_argument("--sync-data", action="store_true")
    parser.add_argument("--force-data-sync", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_data_slice(label: str) -> tuple[int | None, int | None]:
    match = re.fullmatch(r"train(?P<train>all|\d+)_val(?P<val>all|\d+)", label)
    if not match:
        raise SystemExit(f"Unsupported data slice format: {label}")
    train_raw = match.group("train")
    val_raw = match.group("val")
    return (None if train_raw == "all" else int(train_raw), None if val_raw == "all" else int(val_raw))


def parse_schedule(label: str) -> tuple[int, int]:
    match = re.fullmatch(r"lambda_warmup_(?P<warmup>\d+)_of_(?P<epochs>\d+)", label)
    if not match:
        raise SystemExit(f"Unsupported schedule format: {label}")
    return int(match.group("warmup")), int(match.group("epochs"))


def kernel_mix_label(body_kernel_size: int | None, alt_kernel_size: int | None) -> str:
    if body_kernel_size and alt_kernel_size and body_kernel_size != alt_kernel_size:
        return f"{body_kernel_size}x{body_kernel_size}+{alt_kernel_size}x{alt_kernel_size}"
    if body_kernel_size:
        return f"{body_kernel_size}x{body_kernel_size}-only"
    return "unknown"


def resolve_candidate_id(args: argparse.Namespace) -> str:
    get_candidate_spec, list_candidate_ids = import_candidate_registry()
    if args.candidate_id:
        get_candidate_spec(args.candidate_id)
        return args.candidate_id

    for candidate_id in list_candidate_ids(include_extreme=True):
        spec = get_candidate_spec(candidate_id)
        if spec.architecture != args.family:
            continue
        if spec.num_blocks != args.depth:
            continue
        if spec.channels != args.width:
            continue
        if kernel_mix_label(spec.body_kernel_size, spec.alt_kernel_size) != args.kernel_mix:
            continue
        return candidate_id

    raise SystemExit(
        "Could not resolve candidate_id from the provided CLI shape. "
        "Pass --candidate-id to target a known registry candidate explicitly."
    )


def comparison_signature(args: argparse.Namespace, *, train_pairs: int | None, val_pairs: int | None, num_epochs: int) -> dict[str, Any]:
    return {
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "num_epochs": num_epochs,
        "batch_size": args.batch_size,
        "eval_size": args.eval_size,
        "train_patch_size": args.train_patch_size,
        "backend": "modal",
        "modal_gpu": args.modal_gpu,
    }


def notebook_env(args: argparse.Namespace, *, candidate_id: str, started_day: str, train_pairs: int | None, val_pairs: int | None, warmup_epochs: int, num_epochs: int) -> dict[str, str]:
    return build_notebook_env(
        {
            "LAB3_NOTEBOOK_CANDIDATE_ID": candidate_id,
            "LAB3_NOTEBOOK_RUN_NAME": args.run_id,
            "LAB3_NOTEBOOK_STARTED_DAY": started_day,
            "LAB3_NOTEBOOK_BATCH_SIZE": args.batch_size,
            "LAB3_NOTEBOOK_NUM_EPOCHS": num_epochs,
            "LAB3_NOTEBOOK_WARMUP_EPOCHS": warmup_epochs,
            "LAB3_NOTEBOOK_EVAL_SIZE": args.eval_size,
            "LAB3_NOTEBOOK_TRAIN_PATCH": args.train_patch_size,
            "LAB3_NOTEBOOK_TRAIN_PAIR_LIMIT": train_pairs,
            "LAB3_NOTEBOOK_VAL_PAIR_LIMIT": val_pairs,
            "LAB3_NOTEBOOK_MODAL_TIMEOUT_MINUTES": args.budget_minutes,
            "LAB3_NOTEBOOK_POLL_INTERVAL_MINUTES": args.poll_interval_minutes or min(30, max(1, args.budget_minutes)),
            "LAB3_NOTEBOOK_MODAL_DATA_VOLUME": args.modal_data_volume,
            "LAB3_NOTEBOOK_MODAL_RUNS_VOLUME": args.modal_runs_volume,
            "LAB3_NOTEBOOK_PRIOR_BEST_VAL_PSNR": args.prior_best_val_psnr,
            "LAB3_MODAL_GPU": args.modal_gpu,
            "LAB3_NOTEBOOK_SYNC_DATA": args.sync_data or args.force_data_sync,
            "LAB3_NOTEBOOK_FORCE_DATA_SYNC": args.force_data_sync,
        }
    )


def failure_payload(
    *,
    args: argparse.Namespace,
    candidate_id: str,
    run_root: Path,
    executed_notebook_path: Path,
    message: str,
    comparison_basis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "candidate_id": candidate_id,
        "run_root": str(run_root),
        "summary_path": None,
        "report_path": None,
        "notebook_path": None,
        "executed_notebook_path": str(executed_notebook_path),
        "status": "launcher_failed",
        "completed": False,
        "cut_off": False,
        "validation_psnr": None,
        "delta_psnr": None,
        "input_psnr": None,
        "val_loss": None,
        "comparison_signature": comparison_basis,
        "artifact_readiness": {
            "pth_ready": False,
            "onnx_ready": False,
            "calibration_ready": False,
            "mxq_handoff_ready": False,
        },
        "onnx_sanity": {},
        "calibration": {},
        "mxq_handoff": {},
        "promotion_gates": {},
        "notes": [message],
    }


def main() -> None:
    args = parse_args()
    if args.activation != "leaky_relu":
        raise SystemExit("Unsupported activation for the canonical notebook path; expected leaky_relu")
    if args.optimizer != "adamw":
        raise SystemExit("Unsupported optimizer for the canonical notebook path; expected adamw")

    candidate_id = resolve_candidate_id(args)
    train_pairs, val_pairs = parse_data_slice(args.data_slice)
    warmup_epochs, num_epochs = parse_schedule(args.schedule)
    started_day = time.strftime("%Y-%m-%d")
    run_root = PROJECT_ROOT / "runs" / started_day / args.run_id
    executed_notebook_path = run_root / "notebooks" / "lab3_wide_residual_nobn_modal_app_executed.ipynb"
    env = notebook_env(
        args,
        candidate_id=candidate_id,
        started_day=started_day,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        warmup_epochs=warmup_epochs,
        num_epochs=num_epochs,
    )

    try:
        payload = execute_notebook(
            notebook_path=CANONICAL_NOTEBOOK,
            output_path=executed_notebook_path,
            working_dir=PROJECT_ROOT,
            env_overrides=env,
        )
    except NotebookExecutionFailure as exc:
        payload = failure_payload(
            args=args,
            candidate_id=candidate_id,
            run_root=run_root,
            executed_notebook_path=exc.output_path,
            message=str(exc),
            comparison_basis=comparison_signature(
                args,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                num_epochs=num_epochs,
            ),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1) from exc

    payload["executed_notebook_path"] = str(executed_notebook_path)
    payload["run_id"] = args.run_id
    payload["candidate_id"] = payload.get("candidate_id", candidate_id)
    payload["comparison_signature"] = comparison_signature(
        args,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        num_epochs=num_epochs,
    )
    launcher_result_path = Path(payload["run_root"]) / "launcher_result.json"
    write_json(launcher_result_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
