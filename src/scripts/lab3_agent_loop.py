from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from generate_lab3_notebook import write_notebook
from lab3_modal_app import (
    DEFAULT_DATA_VOLUME_NAME,
    DEFAULT_GPU,
    DEFAULT_POLL_INTERVAL_MINUTES,
    DEFAULT_RUNS_VOLUME_NAME,
    DEFAULT_TIMEOUT_MINUTES,
    execute_modal_pipeline,
)
from lab3_pipeline_lib import (
    PipelineConfig,
    best_val_psnr_for_signature,
    comparison_signature_from_cfg,
    comparison_signature_from_summary,
    default_data_root,
    get_candidate_spec,
    list_candidate_ids,
    load_run_summaries,
    run_layout_from_config,
    run_pipeline,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded Codex autopilot for Lab 3 notebook experiments.")
    parser.add_argument("--budget", type=int, default=1, help="Number of experiment candidates to process.")
    parser.add_argument("--dry-run", action="store_true", help="Plan candidates and generate notebooks without training.")
    parser.add_argument("--force-candidate", default="", help="Candidate id to force for every budget slot.")
    parser.add_argument("--allow-extreme", action="store_true", help="Allow explicitly whitelisted extreme candidates.")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-pair-limit", type=int, default=None)
    parser.add_argument("--val-pair-limit", type=int, default=None)
    parser.add_argument("--run-name-prefix", default="autopilot")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--modal-data-volume", default=DEFAULT_DATA_VOLUME_NAME)
    parser.add_argument("--modal-runs-volume", default=DEFAULT_RUNS_VOLUME_NAME)
    parser.add_argument("--modal-gpu", default=os.environ.get("LAB3_MODAL_GPU", DEFAULT_GPU))
    parser.add_argument("--modal-timeout-minutes", type=int, default=DEFAULT_TIMEOUT_MINUTES)
    parser.add_argument("--modal-poll-interval-minutes", type=int, default=DEFAULT_POLL_INTERVAL_MINUTES)
    parser.add_argument("--sync-data", action="store_true", help="Sync local Data/ into the Modal data volume before runs.")
    parser.add_argument("--force-data-sync", action="store_true", help="Force-refresh the Modal data volume before runs.")
    return parser.parse_args()


def notebook_variant_for_backend(backend: str) -> str:
    return "modal" if backend == "modal" else "local"


def candidate_distance(candidate_id: str, reference_id: str) -> tuple[int, int]:
    candidate = get_candidate_spec(candidate_id)
    reference = get_candidate_spec(reference_id)
    return (
        abs(candidate.channels - reference.channels),
        abs(candidate.num_blocks - reference.num_blocks),
    )


def promoted_summaries(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [summary for summary in history if summary.get("gates", {}).get("promotion_pass")]


def best_summary(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = promoted_summaries(history)
    if not ranked:
        ranked = [summary for summary in history if summary.get("evaluation", {}).get("val_psnr") is not None]
    if not ranked:
        return None
    return max(ranked, key=lambda summary: summary.get("evaluation", {}).get("val_psnr", float("-inf")))


def best_val_psnr(history: list[dict[str, Any]]) -> float | None:
    best = best_summary(history)
    if not best:
        return None
    return best.get("evaluation", {}).get("val_psnr")


def best_summary_for_signature(history: list[dict[str, Any]], signature: dict[str, Any]) -> dict[str, Any] | None:
    comparable = [
        summary for summary in history
        if summary.get("evaluation", {}).get("val_psnr") is not None
        and comparison_signature_from_summary(summary) == signature
    ]
    if not comparable:
        return None
    return max(comparable, key=lambda summary: summary.get("evaluation", {}).get("val_psnr", float("-inf")))


def did_recent_runs_improve(history: list[dict[str, Any]]) -> bool:
    promoted = promoted_summaries(history)
    if len(promoted) < 2:
        return False
    promoted = promoted[-2:]
    previous = promoted[0].get("evaluation", {}).get("val_psnr", float("-inf"))
    current = promoted[1].get("evaluation", {}).get("val_psnr", float("-inf"))
    return current > previous


def is_executed_summary(summary: dict[str, Any]) -> bool:
    execution = summary.get("execution", {})
    final_status = execution.get("final_status")
    if final_status in {"planned", "incomplete"}:
        return False
    evaluation = summary.get("evaluation", {})
    return any(
        evaluation.get(key) is not None
        for key in ["val_psnr", "delta_psnr", "input_psnr"]
    )


def choose_next_candidate(
    history: list[dict[str, Any]],
    allow_extreme: bool,
    force_candidate: str = "",
) -> tuple[str | None, str]:
    if force_candidate:
        return force_candidate, "forced candidate requested by caller"

    tried = {
        summary.get("candidate", {}).get("candidate_id")
        for summary in history
        if summary.get("candidate") and is_executed_summary(summary)
    }
    best = best_summary(history)
    best_candidate_id = best.get("candidate", {}).get("candidate_id") if best else "wide_residual_nobn_v1"
    bounded_candidates = list_candidate_ids(include_extreme=False)
    extreme_candidates = [candidate_id for candidate_id in list_candidate_ids(include_extreme=True) if candidate_id not in bounded_candidates]

    if not history:
        return "wide_residual_nobn_v1", "no history yet; start from the baseline candidate"

    if did_recent_runs_improve(history):
        neighbors = sorted(
            [candidate_id for candidate_id in bounded_candidates if candidate_id not in tried],
            key=lambda candidate_id: candidate_distance(candidate_id, best_candidate_id),
        )
        if neighbors:
            return neighbors[0], f"recent promoted run improved; exploit near best candidate {best_candidate_id}"

    for candidate_id in bounded_candidates:
        if candidate_id not in tried:
            return candidate_id, "broaden within the bounded no-BN residual search space"

    if allow_extreme:
        for candidate_id in extreme_candidates:
            if candidate_id not in tried:
                return candidate_id, "bounded candidates exhausted; try the explicit extreme whitelist"

    return None, "no unseen approved candidate remains; broaden the registry or add a new architecture/config before running again"


def autopilot_reports_dir(project_root: Path) -> Path:
    path = project_root / "runs" / "autopilot_reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_cfg(
    project_root: Path,
    args: argparse.Namespace,
    candidate_id: str,
    slot_index: int,
) -> PipelineConfig:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name_prefix}_{slot_index + 1:02d}_{candidate_id}_{timestamp}"
    cfg = PipelineConfig(
        project_root=project_root,
        data_root=default_data_root(project_root),
        artifact_root=project_root,
        run_name=run_name,
        candidate_id=candidate_id,
        backend=args.backend,
        started_day=time.strftime("%Y-%m-%d"),
    )
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.train_pair_limit is not None:
        cfg.train_pair_limit = args.train_pair_limit
    if args.val_pair_limit is not None:
        cfg.val_pair_limit = args.val_pair_limit
    if args.backend == "modal":
        cfg.modal_app_name = "lab3-modal-pipeline"
        cfg.modal_function_name = "run_lab3_pipeline"
        cfg.modal_gpu = args.modal_gpu
        cfg.modal_timeout_minutes = args.modal_timeout_minutes
        cfg.modal_data_volume = args.modal_data_volume
        cfg.modal_runs_volume = args.modal_runs_volume
    return cfg


def summarize_result(summary: dict[str, Any]) -> dict[str, Any]:
    evaluation = summary.get("evaluation", {})
    execution = summary.get("execution", {})
    follow_up = summary.get("follow_up_recommendation", {})
    return {
        "run_day": summary.get("run_day"),
        "run_root": summary.get("run_root"),
        "candidate_id": summary.get("candidate", {}).get("candidate_id"),
        "backend": summary.get("backend", execution.get("backend")),
        "val_psnr": evaluation.get("val_psnr"),
        "delta_psnr": evaluation.get("delta_psnr"),
        "promotion_pass": summary.get("gates", {}).get("promotion_pass"),
        "final_status": execution.get("final_status"),
        "fuller_run_justified": follow_up.get("fuller_run_justified"),
        "notebook_path": summary.get("notebook_path"),
        "summary_path": summary.get("summary_path"),
    }


def materialize_dry_run(cfg: PipelineConfig, reason: str) -> dict[str, Any]:
    candidate = cfg.candidate()
    layout = run_layout_from_config(cfg)
    write_notebook(
        layout.notebook_path,
        cfg.candidate_id,
        run_name=cfg.run_name,
        variant=notebook_variant_for_backend(cfg.backend),
    )
    save_json(layout.config_path, cfg.as_json())
    save_json(
        layout.status_path,
        {
            "phase": "planned",
            "candidate_id": cfg.candidate_id,
            "reason": reason,
            "backend": cfg.backend,
            "run_day": layout.run_day,
            "notebook_path": str(layout.notebook_path),
        },
    )
    summary = {
        "backend": cfg.backend,
        "run_day": layout.run_day,
        "run_root": str(layout.run_root),
        "candidate": candidate.as_json(),
        "reason": reason,
        "notebook_path": str(layout.notebook_path),
        "summary_path": str(layout.summary_path),
        "report_path": str(layout.report_path),
        "evaluation": {},
        "gates": {"promotion_pass": False, "screening_pass": False},
        "planned_only": True,
        "execution": {
            "backend": cfg.backend,
            "final_status": "planned",
            "modal_timeout_minutes": cfg.modal_timeout_minutes,
            "modal_gpu": cfg.modal_gpu,
        },
    }
    save_json(layout.summary_path, {"phase": "planned", "config": cfg.as_json(), "summary": summary})
    save_json(layout.report_path, summary)
    return summary


def next_recommendation(history: list[dict[str, Any]], allow_extreme: bool) -> dict[str, str | None]:
    candidate_id, reason = choose_next_candidate(history, allow_extreme=allow_extreme)
    return {"candidate_id": candidate_id, "reason": reason}


def write_autopilot_report(project_root: Path, report_payload: dict[str, Any]) -> Path:
    reports_dir = autopilot_reports_dir(project_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"autopilot_{timestamp}.json"
    latest_path = reports_dir / "latest.json"
    markdown_path = reports_dir / "latest.md"
    save_json(report_path, report_payload)
    save_json(latest_path, report_payload)

    best = report_payload.get("best_run") or {}
    lines = [
        "# Lab 3 Autopilot Report",
        "",
        f"- Timestamp: {report_payload['timestamp']}",
        f"- Backend: {report_payload['backend']}",
        f"- Best run id: {best.get('run_root', 'n/a')}",
        f"- Best val PSNR: {best.get('val_psnr', 'n/a')}",
        f"- Best delta PSNR: {best.get('delta_psnr', 'n/a')}",
        f"- Best status: {best.get('final_status', 'n/a')}",
        f"- Next recommended candidate: {report_payload['next_recommended_mutation'].get('candidate_id') or 'n/a'}",
        "",
        "## Executed",
    ]
    for item in report_payload["executed_runs"]:
        lines.append(
            f"- {item.get('candidate_id')}: status={item.get('final_status')} "
            f"val_psnr={item.get('val_psnr')} delta={item.get('delta_psnr')} "
            f"promotion_pass={item.get('promotion_pass')} fuller_run_justified={item.get('fuller_run_justified')}"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def execute_candidate(cfg: PipelineConfig, args: argparse.Namespace, prior_best: float | None, reason: str) -> dict[str, Any]:
    layout = run_layout_from_config(cfg)
    write_notebook(
        layout.notebook_path,
        cfg.candidate_id,
        run_name=cfg.run_name,
        variant=notebook_variant_for_backend(cfg.backend),
    )

    if cfg.backend == "modal":
        summary = execute_modal_pipeline(
            cfg,
            prior_best_val_psnr=prior_best,
            sync_data=args.sync_data or args.force_data_sync,
            force_data_sync=args.force_data_sync,
            poll_interval_minutes=args.modal_poll_interval_minutes,
            timeout_minutes=args.modal_timeout_minutes,
            gpu=args.modal_gpu,
            data_volume_name=args.modal_data_volume,
            runs_volume_name=args.modal_runs_volume,
        )
    else:
        summary = run_pipeline(cfg, prior_best_val_psnr=prior_best)

    summary["selection_reason"] = reason
    if summary.get("report_path"):
        save_json(Path(summary["report_path"]), summary)
    return summary


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    history = load_run_summaries(project_root)
    executed: list[dict[str, Any]] = []

    for slot_index in range(max(1, args.budget)):
        candidate_id, reason = choose_next_candidate(
            history + executed,
            allow_extreme=args.allow_extreme,
            force_candidate=args.force_candidate,
        )
        if candidate_id is None:
            break
        cfg = build_cfg(project_root, args, candidate_id, slot_index)

        if args.dry_run:
            summary = materialize_dry_run(cfg, reason)
        else:
            prior_best = best_val_psnr_for_signature(history + executed, comparison_signature_from_cfg(cfg))
            summary = execute_candidate(cfg, args, prior_best, reason)

        executed.append(summary)

    combined_history = history + executed
    best = best_summary(combined_history)
    kept = [summarize_result(summary) for summary in executed if summary.get("gates", {}).get("promotion_pass")]
    discarded = [summarize_result(summary) for summary in executed if not summary.get("gates", {}).get("promotion_pass")]
    recommendation = next_recommendation(combined_history, allow_extreme=args.allow_extreme)
    comparable_basis = comparison_signature_from_cfg(build_cfg(project_root, args, executed[0].get("candidate", {}).get("candidate_id", "wide_residual_nobn_v1"), 0)) if executed else {}
    current_best_comparable = best_summary_for_signature(combined_history, comparable_basis) if comparable_basis else None

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "backend": args.backend,
        "dry_run": args.dry_run,
        "budget": args.budget,
        "executed_runs": [summarize_result(summary) for summary in executed],
        "kept_candidates": kept,
        "discarded_candidates": discarded,
        "best_run": summarize_result(best) if best else {},
        "current_best_comparable_run": summarize_result(current_best_comparable) if current_best_comparable else {},
        "comparison_basis": comparable_basis,
        "best_artifact_paths": {
            "best_checkpoint": best.get("best_checkpoint") if best else None,
            "best_onnx": best.get("onnx_path") if best else None,
            "best_mxq": best.get("mxq_path") if best else None,
            "calibration_dir": best.get("calibration", {}).get("calibration_dir") if best else None,
            "summary_path": best.get("summary_path") if best else None,
            "notebook_path": best.get("notebook_path") if best else None,
        },
        "next_recommended_mutation": recommendation,
    }
    report_path = write_autopilot_report(project_root, report_payload)
    print(json.dumps({"report_path": str(report_path), **report_payload}, indent=2))


if __name__ == "__main__":
    main()
