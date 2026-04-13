from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from lab3_pipeline_lib import default_data_root, default_run_name, get_candidate_spec


def md_cell(text: str) -> dict[str, object]:
    text = textwrap.dedent(text).strip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def code_cell(text: str) -> dict[str, object]:
    text = textwrap.dedent(text).strip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


def helper_code() -> str:
    helper_path = Path(__file__).with_name("lab3_pipeline_lib.py")
    return helper_path.read_text(encoding="utf-8").strip() + "\n"


def modal_execution_code() -> str:
    return """
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from lab3_modal_app import execute_modal_pipeline
    from lab3_pipeline_lib import best_val_psnr_for_signature, comparison_signature_from_cfg, load_run_summaries

    prior_best_val_psnr = best_val_psnr_for_signature(
        load_run_summaries(PROJECT_ROOT),
        comparison_signature_from_cfg(cfg),
    )

    summary = execute_modal_pipeline(
        cfg,
        prior_best_val_psnr=prior_best_val_psnr,
        sync_data=SYNC_DATA_TO_VOLUME,
        force_data_sync=FORCE_DATA_SYNC,
        poll_interval_minutes=POLL_INTERVAL_MINUTES,
        timeout_minutes=MODAL_TIMEOUT_MINUTES,
        gpu=MODAL_GPU,
        data_volume_name=MODAL_DATA_VOLUME,
        runs_volume_name=MODAL_RUNS_VOLUME,
    )
    print_artifact_summary(summary)
    """


def build_notebook(candidate_id: str, run_name: str | None = None, variant: str = "local") -> dict[str, object]:
    candidate = get_candidate_spec(candidate_id)
    notebook_run_name = run_name or default_run_name(candidate_id)
    is_modal = variant == "modal"
    title_suffix = " (Modal App)" if is_modal else ""
    execution_title = "remote Modal app" if is_modal else "submission-shaped NPU-first baseline"
    setup_note = (
        "This notebook submits the pipeline to Modal and syncs the resulting day-partitioned run directory back locally."
        if is_modal
        else "This notebook is fully self-contained. It defines the model, data pipeline, training loop, validation, ONNX export, training-derived calibration export, and MXQ handoff metadata in one place."
    )
    execution_cell = modal_execution_code() if is_modal else """
    summary = run_pipeline(cfg)
    print_artifact_summary(summary)
    """
    cells = [
        md_cell(
            f"""
            # Lab 3 - {candidate.candidate_id}{title_suffix}

            - Goal: build a {execution_title} for Lab 3
            - Candidate: `{candidate.candidate_id}`
            - Model family: same-resolution no-BN residual CNN
            - Expected input/output: `256x256x3`
            - Search tier: `{candidate.search_tier}`
            """
        ),
        md_cell(
            """
            ## 1. Setup

            This notebook uses the Lab 3 rubric structure while keeping the operational execution path explicit.
            """
        ),
        md_cell(
            f"""
            {setup_note}
            """
        ),
        code_cell(helper_code()),
        code_cell(
            f"""
            PROJECT_ROOT = Path.cwd().resolve()
            CANDIDATE_ID = "{candidate.candidate_id}"
            RUN_NAME = os.environ.get("LAB3_NOTEBOOK_RUN_NAME", default_run_name(CANDIDATE_ID))
            DATA_ROOT = default_data_root(PROJECT_ROOT)
            ARTIFACT_ROOT = PROJECT_ROOT
            BACKEND = {"\"modal\"" if is_modal else "\"local\""}

            BATCH_SIZE = int(os.environ.get("LAB3_NOTEBOOK_BATCH_SIZE", "24"))
            NUM_EPOCHS = int(os.environ.get("LAB3_NOTEBOOK_NUM_EPOCHS", "80"))
            LEARNING_RATE = float(os.environ.get("LAB3_NOTEBOOK_LR", "3e-4"))
            WEIGHT_DECAY = float(os.environ.get("LAB3_NOTEBOOK_WEIGHT_DECAY", "2e-4"))
            TRAIN_PATCH_SIZE = int(os.environ.get("LAB3_NOTEBOOK_TRAIN_PATCH", "224"))
            EVAL_SIZE = int(os.environ.get("LAB3_NOTEBOOK_EVAL_SIZE", "256"))
            WARMUP_EPOCHS = int(os.environ.get("LAB3_NOTEBOOK_WARMUP_EPOCHS", "5"))
            EMA_DECAY = float(os.environ.get("LAB3_NOTEBOOK_EMA_DECAY", "0.999"))
            CALIBRATION_COUNT = int(os.environ.get("LAB3_NOTEBOOK_CALIBRATION_COUNT", "128"))

            RUN_TRAINING = True
            RUN_ONNX_EXPORT = True
            VERIFY_ONNX_EXPORT = True
            RUN_MXQ_COMPILE = False

            TRAIN_PAIR_LIMIT = int(os.environ["LAB3_NOTEBOOK_TRAIN_PAIR_LIMIT"]) if os.environ.get("LAB3_NOTEBOOK_TRAIN_PAIR_LIMIT") else None
            VAL_PAIR_LIMIT = int(os.environ["LAB3_NOTEBOOK_VAL_PAIR_LIMIT"]) if os.environ.get("LAB3_NOTEBOOK_VAL_PAIR_LIMIT") else None
            STARTED_DAY = time.strftime("%Y-%m-%d")

            MODAL_DATA_VOLUME = os.environ.get("LAB3_NOTEBOOK_MODAL_DATA_VOLUME", "lab3-data")
            MODAL_RUNS_VOLUME = os.environ.get("LAB3_NOTEBOOK_MODAL_RUNS_VOLUME", "lab3-runs")
            MODAL_GPU = os.environ.get("LAB3_MODAL_GPU", "L40S")
            MODAL_TIMEOUT_MINUTES = int(os.environ.get("LAB3_NOTEBOOK_MODAL_TIMEOUT_MINUTES", "120"))
            POLL_INTERVAL_MINUTES = int(os.environ.get("LAB3_NOTEBOOK_POLL_INTERVAL_MINUTES", "30"))
            SYNC_DATA_TO_VOLUME = os.environ.get("LAB3_NOTEBOOK_SYNC_DATA", "{str(is_modal)}").lower() in {{"1", "true", "yes"}}
            FORCE_DATA_SYNC = os.environ.get("LAB3_NOTEBOOK_FORCE_DATA_SYNC", "false").lower() in {"1", "true", "yes"}

            cfg = PipelineConfig(
                project_root=PROJECT_ROOT,
                data_root=DATA_ROOT,
                run_name=RUN_NAME,
                artifact_root=ARTIFACT_ROOT,
                candidate_id=CANDIDATE_ID,
                batch_size=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                train_patch_size=TRAIN_PATCH_SIZE,
                eval_size=EVAL_SIZE,
                warmup_epochs=WARMUP_EPOCHS,
                ema_decay=EMA_DECAY,
                calibration_count=CALIBRATION_COUNT,
                run_training=RUN_TRAINING,
                run_onnx_export=RUN_ONNX_EXPORT,
                verify_onnx_export=VERIFY_ONNX_EXPORT,
                run_mxq_compile=RUN_MXQ_COMPILE,
                train_pair_limit=TRAIN_PAIR_LIMIT,
                val_pair_limit=VAL_PAIR_LIMIT,
                backend=BACKEND,
                started_day=STARTED_DAY,
            )

            print(json.dumps(cfg.as_json(), indent=2))
            """
        ),
        md_cell("## 2. Data"),
        code_cell(
            """
            set_seed(cfg.seed)
            device = resolve_device()
            train_pairs = collect_train_pairs(cfg.data_root, cfg.train_pair_limit)
            val_pairs = collect_val_pairs(cfg.data_root, cfg.val_pair_limit)
            if not train_pairs:
                raise FileNotFoundError(f"No training pairs found under {cfg.data_root}")
            if not val_pairs:
                raise FileNotFoundError(f"No validation pairs found under {cfg.data_root}")

            pair_summary = summarize_pairs(train_pairs, val_pairs)
            print(f"Using device: {device}")
            print(json.dumps(pair_summary, indent=2))

            train_loader, val_loader = make_dataloaders(train_pairs, val_pairs, cfg, device)
            lr_batch, hr_batch, names = next(iter(val_loader))
            print("Validation batch names:", list(names[:3]))
            print("Validation batch input shape:", tuple(lr_batch.shape))
            print("Validation batch target shape:", tuple(hr_batch.shape))
            """
        ),
        md_cell("## 3. Model and Training"),
        code_cell(
            """
            model = build_model(cfg).to(device)
            contract = verify_model_contract(model, cfg.eval_size)
            verify_residual_l1_batch(model, train_loader, device, cfg)

            print("Model contract:")
            print(json.dumps(contract, indent=2))
            print("Operator audit:")
            print(json.dumps(operator_audit(model), indent=2))
            """
        ),
        code_cell(
            execution_cell
        ),
        md_cell("## 4. Validation and ONNX"),
        code_cell(
            """
            print("Evaluation summary:")
            print(json.dumps(summary["evaluation"], indent=2))
            print("ONNX export details:")
            print(json.dumps(summary["onnx"], indent=2))
            """
        ),
        md_cell("## 5. Calibration and MXQ Handoff"),
        code_cell(
            """
            print("Calibration details:")
            print(json.dumps(summary["calibration"], indent=2))
            print("MXQ handoff details:")
            print(json.dumps(summary["mxq"], indent=2))
            """
        ),
        md_cell("## 6. Final Artifact Summary"),
        code_cell(
            """
            final_paths = {
                "best_checkpoint": summary["best_checkpoint"],
                "best_onnx": summary["onnx_path"],
                "calibration_dir": summary["calibration"]["calibration_dir"],
                "mxq_path": summary["mxq_path"],
                "run_summary": summary["summary_path"],
                "run_report": summary["report_path"],
                "run_day": summary.get("run_day"),
                "synced_local_run_root": summary.get("synced_local_run_root"),
            }
            print(json.dumps(final_paths, indent=2))
            """
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def default_output_path(project_root: Path, candidate_id: str, variant: str = "local") -> Path:
    candidate = get_candidate_spec(candidate_id)
    if variant == "modal" and candidate_id == "wide_residual_nobn_v1":
        return project_root / "lab3_wide_residual_nobn_modal_app.ipynb"
    if variant == "local" and candidate_id == "wide_residual_nobn_v1":
        return project_root / "lab3_wide_residual_nobn_full_pipeline.ipynb"
    suffix = "modal_app" if variant == "modal" else "full_pipeline"
    return project_root / f"lab3_{candidate.notebook_slug}_{suffix}.ipynb"


def write_notebook(output_path: Path, candidate_id: str, run_name: str | None = None, variant: str = "local") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_notebook(candidate_id, run_name=run_name, variant=variant), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a submission-shaped Lab 3 notebook for a candidate.")
    parser.add_argument("--candidate-id", default="wide_residual_nobn_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--variant", choices=["local", "modal"], default="local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_path = args.output or default_output_path(project_root, args.candidate_id, variant=args.variant)
    notebook_path = write_notebook(output_path, args.candidate_id, run_name=args.run_name or None, variant=args.variant)
    print(notebook_path)


if __name__ == "__main__":
    main()
