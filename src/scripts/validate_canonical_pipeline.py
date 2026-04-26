#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from notebook_execution import NotebookExecutionFailure, build_notebook_env, execute_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the canonical Lab 3 Modal notebook as a bounded validation smoke test.")
    parser.add_argument("--notebook", required=True, help="Path to the canonical notebook.")
    parser.add_argument("--candidate-id", default="wide_residual_nobn_v1")
    parser.add_argument("--train-pairs", type=int, default=8)
    parser.add_argument("--val-pairs", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--train-patch-size", type=int, default=224)
    parser.add_argument("--budget-minutes", type=int, default=10)
    parser.add_argument("--modal-gpu", default="L40S")
    parser.add_argument("--modal-data-volume", default="lab3-data")
    parser.add_argument("--modal-runs-volume", default="lab3-runs")
    parser.add_argument("--sync-data", action="store_true")
    parser.add_argument("--force-data-sync", action="store_true")
    return parser.parse_args()


def validate_payload(payload: dict[str, object]) -> None:
    if payload.get("status") != "completed":
        raise SystemExit(f"Canonical notebook validation failed: status={payload.get('status')}")
    for key in ["run_root", "summary_path", "report_path"]:
        if not payload.get(key):
            raise SystemExit(f"Canonical notebook validation failed: missing {key}")
    onnx = payload.get("onnx_sanity", {})
    if not isinstance(onnx, dict) or not onnx.get("passed"):
        raise SystemExit("Canonical notebook validation failed: ONNX sanity did not pass")
    calibration = payload.get("calibration", {})
    if not isinstance(calibration, dict) or not calibration.get("passed") or not calibration.get("derived_from_training"):
        raise SystemExit("Canonical notebook validation failed: calibration output is missing or not training-derived")
    mxq_handoff = payload.get("mxq_handoff", {})
    if not isinstance(mxq_handoff, dict) or not mxq_handoff.get("passed"):
        raise SystemExit("Canonical notebook validation failed: MXQ handoff did not pass")


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook).expanduser().resolve()
    if not notebook_path.exists():
        raise SystemExit(f"Canonical notebook not found: {notebook_path}")

    run_name = f"notebookvalidate_{time.strftime('%Y%m%d_%H%M%S')}"
    output_path = PROJECT_ROOT / "tmp" / "notebook_validation" / f"{run_name}.ipynb"
    env = build_notebook_env(
        {
            "LAB3_NOTEBOOK_CANDIDATE_ID": args.candidate_id,
            "LAB3_NOTEBOOK_RUN_NAME": run_name,
            "LAB3_NOTEBOOK_STARTED_DAY": time.strftime("%Y-%m-%d"),
            "LAB3_NOTEBOOK_BATCH_SIZE": args.batch_size,
            "LAB3_NOTEBOOK_NUM_EPOCHS": args.num_epochs,
            "LAB3_NOTEBOOK_WARMUP_EPOCHS": args.warmup_epochs,
            "LAB3_NOTEBOOK_EVAL_SIZE": args.eval_size,
            "LAB3_NOTEBOOK_TRAIN_PATCH": args.train_patch_size,
            "LAB3_NOTEBOOK_TRAIN_PAIR_LIMIT": args.train_pairs,
            "LAB3_NOTEBOOK_VAL_PAIR_LIMIT": args.val_pairs,
            "LAB3_NOTEBOOK_MODAL_TIMEOUT_MINUTES": args.budget_minutes,
            "LAB3_NOTEBOOK_POLL_INTERVAL_MINUTES": min(2, max(1, args.budget_minutes)),
            "LAB3_NOTEBOOK_MODAL_DATA_VOLUME": args.modal_data_volume,
            "LAB3_NOTEBOOK_MODAL_RUNS_VOLUME": args.modal_runs_volume,
            "LAB3_MODAL_GPU": args.modal_gpu,
            "LAB3_NOTEBOOK_SYNC_DATA": args.sync_data or args.force_data_sync,
            "LAB3_NOTEBOOK_FORCE_DATA_SYNC": args.force_data_sync,
        }
    )

    try:
        payload = execute_notebook(
            notebook_path=notebook_path,
            output_path=output_path,
            working_dir=PROJECT_ROOT,
            env_overrides=env,
        )
    except NotebookExecutionFailure as exc:
        raise SystemExit(f"{exc}\nExecuted notebook artifact: {exc.output_path}") from exc

    validate_payload(payload)
    print(
        json.dumps(
            {
                "status": "validated",
                "candidate_id": payload["candidate_id"],
                "run_root": payload["run_root"],
                "summary_path": payload["summary_path"],
                "report_path": payload["report_path"],
                "executed_notebook_path": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
