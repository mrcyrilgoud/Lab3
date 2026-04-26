#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NOTEBOOK = PROJECT_ROOT / "lab3_wide_residual_nobn_modal_app.ipynb"
REPORTS_DIR = PROJECT_ROOT / "runs" / "autopilot_reports"
LEDGER_PATH = REPORTS_DIR / "ledger.jsonl"
BEST_KNOWN_PATH = REPORTS_DIR / "best_known.json"
INBOX_SUMMARY_PATH = REPORTS_DIR / "inbox_summary.md"
VALIDATE_SCRIPT = PROJECT_ROOT / "scripts" / "validate_canonical_pipeline.py"
LAUNCH_SCRIPT = PROJECT_ROOT / "scripts" / "run_modal_experiment.py"


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    family: str
    depth: int
    width: int
    kernel_mix: str
    activation: str
    optimizer: str
    schedule: str
    data_slice: str
    budget_minutes: int
    batch_size: int
    eval_size: int
    train_patch_size: int
    modal_gpu: str
    run_category: str
    rerun_reason: str | None = None

    def duplicate_key(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "depth": self.depth,
            "width": self.width,
            "kernel_mix": self.kernel_mix,
            "activation": self.activation,
            "optimizer": self.optimizer,
            "schedule": self.schedule,
            "data_slice": self.data_slice,
            "budget_minutes": self.budget_minutes,
            "batch_size": self.batch_size,
            "eval_size": self.eval_size,
            "train_patch_size": self.train_patch_size,
            "modal_gpu": self.modal_gpu,
        }

    def comparison_signature(self) -> dict[str, Any]:
        warmup_epochs, num_epochs = parse_schedule(self.schedule)
        train_pairs, val_pairs = parse_data_slice(self.data_slice)
        return {
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "num_epochs": num_epochs,
            "batch_size": self.batch_size,
            "eval_size": self.eval_size,
            "train_patch_size": self.train_patch_size,
            "backend": "modal",
            "modal_gpu": self.modal_gpu,
        }


@dataclass
class RunResult:
    run_id: str
    config_hash: str
    source: str
    status: str
    completed: bool
    cut_off: bool
    run_category: str
    candidate: dict[str, Any]
    run_root: str
    summary_path: str | None
    report_path: str | None
    notebook_path: str | None
    started_at: str
    ended_at: str
    wall_clock_minutes: float | None
    validation_psnr: float | None
    delta_psnr: float | None
    input_psnr: float | None
    val_loss: float | None
    artifact_readiness: dict[str, bool]
    comparison_signature: dict[str, Any]
    onnx_sanity: dict[str, Any]
    calibration: dict[str, Any]
    mxq_handoff: dict[str, Any]
    promotion_gates: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    rerun_reason: str | None = None

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ledger-driven Lab 3 autopilot controller.")
    parser.add_argument("--canonical-notebook", default=str(CANONICAL_NOTEBOOK))
    parser.add_argument("--total-budget-hours", type=float, default=4.0)
    parser.add_argument("--checkpoint-minutes", type=int, default=30)
    parser.add_argument("--budget-minutes-per-run", type=int, default=25)
    parser.add_argument("--train-pairs", type=int, default=512)
    parser.add_argument("--val-pairs", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--train-patch-size", type=int, default=224)
    parser.add_argument("--modal-gpu", default="L40S")
    parser.add_argument("--modal-data-volume", default="lab3-data")
    parser.add_argument("--modal-runs-volume", default="lab3-runs")
    parser.add_argument("--rerun-reason", default="")
    parser.add_argument("--force-candidate", default="")
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--allow-extreme", action="store_true")
    parser.add_argument("--sync-data", action="store_true")
    parser.add_argument("--force-data-sync", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not LEDGER_PATH.exists():
        LEDGER_PATH.write_text("", encoding="utf-8")
    if not BEST_KNOWN_PATH.exists():
        write_json(BEST_KNOWN_PATH, empty_best_known(target_signature={}))
    if not INBOX_SUMMARY_PATH.exists():
        INBOX_SUMMARY_PATH.write_text("# Lab 3 Autopilot Inbox Summary\n", encoding="utf-8")


def import_pipeline_helpers() -> tuple[Any, Any]:
    tools_dir = PROJECT_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    from lab3_pipeline_lib import get_candidate_spec, list_candidate_ids  # type: ignore

    return get_candidate_spec, list_candidate_ids


def parse_data_slice(label: str) -> tuple[int | None, int | None]:
    if not label.startswith("train") or "_val" not in label:
        raise ValueError(f"Unsupported data slice format: {label}")
    train_raw, val_raw = label.removeprefix("train").split("_val", maxsplit=1)
    train_pairs = None if train_raw == "all" else int(train_raw)
    val_pairs = None if val_raw == "all" else int(val_raw)
    return train_pairs, val_pairs


def parse_schedule(label: str) -> tuple[int, int]:
    prefix = "lambda_warmup_"
    if not label.startswith(prefix) or "_of_" not in label:
        raise ValueError(f"Unsupported schedule format: {label}")
    warmup_raw, epochs_raw = label.removeprefix(prefix).split("_of_", maxsplit=1)
    return int(warmup_raw), int(epochs_raw)


def schedule_label(warmup_epochs: int, num_epochs: int) -> str:
    return f"lambda_warmup_{min(warmup_epochs, num_epochs)}_of_{num_epochs}"


def data_slice_label(train_pairs: int | None, val_pairs: int | None) -> str:
    train_text = "all" if train_pairs is None else str(train_pairs)
    val_text = "all" if val_pairs is None else str(val_pairs)
    return f"train{train_text}_val{val_text}"


def kernel_mix_label(body_kernel_size: int | None, alt_kernel_size: int | None) -> str:
    if body_kernel_size and alt_kernel_size and body_kernel_size != alt_kernel_size:
        return f"{body_kernel_size}x{body_kernel_size}+{alt_kernel_size}x{alt_kernel_size}"
    if body_kernel_size:
        return f"{body_kernel_size}x{body_kernel_size}-only"
    return "unknown"


def classify_candidate(candidate_id: str, search_tier: str | None = None) -> str:
    if candidate_id == "wide_residual_nobn_v1":
        return "benchmark"
    if search_tier == "extreme" or candidate_id == "wide_residual_nobn_xwide_deep":
        return "investigation"
    return "exploration"


def stable_config_hash(candidate: CandidateConfig) -> str:
    digest = hashlib.sha256(
        json.dumps(candidate.duplicate_key(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest


def empty_best_known(target_signature: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "no_comparable_benchmark_yet",
        "updated_at": now_iso(),
        "comparison_signature": target_signature,
        "run_id": None,
        "config_hash": None,
        "run_root": None,
        "summary_path": None,
        "report_path": None,
        "run_category": None,
        "validation_psnr": None,
        "delta_psnr": None,
        "completed": False,
        "cut_off": False,
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
        "source": None,
    }


def target_signature_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "eval_size": args.eval_size,
        "train_patch_size": args.train_patch_size,
        "backend": "modal",
        "modal_gpu": args.modal_gpu,
    }


def candidate_from_registry(
    candidate_id: str,
    args: argparse.Namespace,
    run_category: str | None = None,
    rerun_reason: str | None = None,
) -> CandidateConfig:
    get_candidate_spec, _list_candidate_ids = import_pipeline_helpers()
    spec = get_candidate_spec(candidate_id)
    return CandidateConfig(
        candidate_id=spec.candidate_id,
        family=spec.architecture,
        depth=spec.num_blocks,
        width=spec.channels,
        kernel_mix=kernel_mix_label(spec.body_kernel_size, spec.alt_kernel_size),
        activation="leaky_relu",
        optimizer="adamw",
        schedule=schedule_label(args.warmup_epochs, args.num_epochs),
        data_slice=data_slice_label(args.train_pairs, args.val_pairs),
        budget_minutes=args.budget_minutes_per_run,
        batch_size=args.batch_size,
        eval_size=args.eval_size,
        train_patch_size=args.train_patch_size,
        modal_gpu=args.modal_gpu,
        run_category=run_category or classify_candidate(spec.candidate_id, spec.search_tier),
        rerun_reason=rerun_reason or None,
    )


def candidate_distance(candidate_id: str, reference_id: str) -> tuple[int, int]:
    get_candidate_spec, _list_candidate_ids = import_pipeline_helpers()
    candidate = get_candidate_spec(candidate_id)
    reference = get_candidate_spec(reference_id)
    return (
        abs(candidate.channels - reference.channels),
        abs(candidate.num_blocks - reference.num_blocks),
    )


def iter_summary_paths() -> list[Path]:
    runs_root = PROJECT_ROOT / "runs"
    nested = sorted(runs_root.glob("*/*/summary.json"))
    legacy = sorted(runs_root.glob("*/summary.json"))
    return nested + [path for path in legacy if path not in nested]


def load_existing_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for summary_path in iter_summary_paths():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summary = payload.get("summary", payload)
        if not isinstance(summary, dict):
            continue
        config = summary.get("config") or payload.get("config") or {}
        if config and not summary.get("config"):
            summary = {**summary, "config": config}
        summaries.append(summary)
    return summaries


def build_artifact_readiness_from_summary(summary: dict[str, Any]) -> dict[str, bool]:
    gates = summary.get("gates", {})
    onnx = summary.get("onnx", {})
    calibration = summary.get("calibration", {})
    mxq = summary.get("mxq", {})
    best_checkpoint = summary.get("best_checkpoint")
    return {
        "pth_ready": bool(best_checkpoint and Path(best_checkpoint).exists()),
        "onnx_ready": bool(gates.get("onnx_pass") and onnx.get("onnx_path") and Path(onnx["onnx_path"]).exists()),
        "calibration_ready": bool(
            gates.get("calibration_pass")
            and calibration.get("manifest_path")
            and Path(calibration["manifest_path"]).exists()
        ),
        "mxq_handoff_ready": bool(gates.get("mxq_handoff_pass") and mxq.get("status") in {"handoff_only", "dry_run", "completed"}),
    }


def candidate_from_summary(summary: dict[str, Any]) -> CandidateConfig:
    config = summary.get("config", {})
    candidate = summary.get("candidate") or config.get("candidate") or {}
    candidate_id = candidate.get("candidate_id", "unknown")
    family = candidate.get("architecture")
    kernel_mix = kernel_mix_label(candidate.get("body_kernel_size"), candidate.get("alt_kernel_size"))

    if family in {None, "unknown"} or kernel_mix == "unknown":
        try:
            get_candidate_spec, _list_candidate_ids = import_pipeline_helpers()
            spec = get_candidate_spec(candidate_id)
            family = family or spec.architecture
            kernel_mix = kernel_mix_label(spec.body_kernel_size, spec.alt_kernel_size)
        except Exception:
            family = family or "unknown"

    schedule = schedule_label(
        int(config.get("warmup_epochs", 0) or 0),
        int(config.get("num_epochs", 0) or 0),
    )
    return CandidateConfig(
        candidate_id=candidate_id,
        family=family or "unknown",
        depth=int(candidate.get("num_blocks", 0) or 0),
        width=int(candidate.get("channels", 0) or 0),
        kernel_mix=kernel_mix,
        activation="leaky_relu",
        optimizer="adamw",
        schedule=schedule,
        data_slice=data_slice_label(config.get("train_pair_limit"), config.get("val_pair_limit")),
        budget_minutes=int(config.get("modal_timeout_minutes") or 0),
        batch_size=int(config.get("batch_size", 0) or 0),
        eval_size=int(config.get("eval_size", 0) or 0),
        train_patch_size=int(config.get("train_patch_size", 0) or 0),
        modal_gpu=(summary.get("execution", {}) or {}).get("modal_gpu") or config.get("modal_gpu") or "",
        run_category=classify_candidate(candidate_id, candidate.get("search_tier")),
        rerun_reason=None,
    )


def run_result_from_summary(
    summary: dict[str, Any],
    *,
    source: str,
    rerun_reason: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    wall_clock_minutes: float | None = None,
) -> RunResult:
    candidate = candidate_from_summary(summary)
    config_hash = stable_config_hash(candidate)
    evaluation = summary.get("evaluation", {})
    execution = summary.get("execution", {})
    gates = summary.get("gates", {})
    onnx = summary.get("onnx", {})
    calibration = summary.get("calibration", {})
    mxq = summary.get("mxq", {})
    final_status = execution.get("final_status")
    if not final_status:
        final_status = "completed" if evaluation.get("val_psnr") is not None else "imported"
    completed = final_status == "completed"
    cut_off = final_status == "cut_off"
    run_root = summary.get("run_root", "")
    run_id = Path(run_root).name if run_root else f"imported-{config_hash[:8]}"
    return RunResult(
        run_id=run_id,
        config_hash=config_hash,
        source=source,
        status=final_status,
        completed=completed,
        cut_off=cut_off,
        run_category=candidate.run_category,
        candidate={**candidate.duplicate_key(), "run_category": candidate.run_category},
        run_root=run_root,
        summary_path=summary.get("summary_path"),
        report_path=summary.get("report_path"),
        notebook_path=summary.get("notebook_path"),
        started_at=started_at or summary.get("run_day", "unknown"),
        ended_at=ended_at or summary.get("run_day", "unknown"),
        wall_clock_minutes=wall_clock_minutes,
        validation_psnr=evaluation.get("val_psnr"),
        delta_psnr=evaluation.get("delta_psnr"),
        input_psnr=evaluation.get("input_psnr"),
        val_loss=evaluation.get("val_loss"),
        artifact_readiness=build_artifact_readiness_from_summary(summary),
        comparison_signature=candidate.comparison_signature(),
        onnx_sanity={
            "passed": gates.get("onnx_pass", False),
            "checker": onnx.get("onnx_checker"),
            "onnx_path": onnx.get("onnx_path"),
            "ort_max_diff": onnx.get("ort_max_diff"),
            "ort_mean_diff": onnx.get("ort_mean_diff"),
        },
        calibration={
            "passed": gates.get("calibration_pass", False),
            "count": calibration.get("count"),
            "derived_from_training": calibration.get("derived_from_training"),
            "manifest_path": calibration.get("manifest_path"),
            "calibration_dir": calibration.get("calibration_dir"),
        },
        mxq_handoff={
            "passed": gates.get("mxq_handoff_pass", False),
            "status": mxq.get("status"),
            "output": mxq.get("output"),
            "output_exists": mxq.get("output_exists"),
            "helper_path": mxq.get("helper_path"),
        },
        promotion_gates=gates,
        notes=["Imported from an existing Lab 3 run summary."],
        rerun_reason=rerun_reason,
    )


def ledger_needs_refresh(entries: list[dict[str, Any]]) -> bool:
    if not entries:
        return True
    required_keys = {"onnx_sanity", "calibration", "mxq_handoff", "summary_path", "report_path"}
    return any(not required_keys.issubset(entry.keys()) for entry in entries)


def refresh_ledger_from_summaries() -> list[dict[str, Any]]:
    summaries = load_existing_summaries()
    normalized = [
        run_result_from_summary(summary, source="bootstrap").as_json()
        for summary in sorted(summaries, key=lambda item: str(item.get("run_root", "")))
    ]
    LEDGER_PATH.write_text("", encoding="utf-8")
    for record in normalized:
        append_jsonl(LEDGER_PATH, record)
    return normalized


def bootstrap_or_refresh_ledger() -> list[dict[str, Any]]:
    ledger_entries = load_ledger(LEDGER_PATH)
    if ledger_needs_refresh(ledger_entries):
        return refresh_ledger_from_summaries()
    return ledger_entries


def entry_matches_signature(entry: dict[str, Any], target_signature: dict[str, Any]) -> bool:
    signature = entry.get("comparison_signature", {})
    for key, value in target_signature.items():
        if value is None:
            continue
        if signature.get(key) != value:
            return False
    return True


def comparable_entries(ledger_entries: list[dict[str, Any]], target_signature: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in ledger_entries if entry_matches_signature(entry, target_signature)]


def best_val_psnr_for_signature(ledger_entries: list[dict[str, Any]], target_signature: dict[str, Any]) -> float | None:
    candidates = [
        entry.get("validation_psnr")
        for entry in comparable_entries(ledger_entries, target_signature)
        if entry.get("validation_psnr") is not None
    ]
    return max(candidates) if candidates else None


def rebuild_best_known(ledger_entries: list[dict[str, Any]], target_signature: dict[str, Any]) -> dict[str, Any]:
    best_entry: dict[str, Any] | None = None
    for entry in comparable_entries(ledger_entries, target_signature):
        if entry.get("run_category") != "benchmark":
            continue
        if entry.get("validation_psnr") is None:
            continue
        if best_entry is None or entry["validation_psnr"] > best_entry["validation_psnr"]:
            best_entry = entry

    if best_entry is None:
        payload = empty_best_known(target_signature)
        write_json(BEST_KNOWN_PATH, payload)
        return payload

    payload = {
        "status": "ready",
        "updated_at": now_iso(),
        "comparison_signature": target_signature,
        "run_id": best_entry.get("run_id"),
        "config_hash": best_entry.get("config_hash"),
        "run_root": best_entry.get("run_root"),
        "summary_path": best_entry.get("summary_path"),
        "report_path": best_entry.get("report_path"),
        "run_category": best_entry.get("run_category"),
        "validation_psnr": best_entry.get("validation_psnr"),
        "delta_psnr": best_entry.get("delta_psnr"),
        "completed": best_entry.get("completed", False),
        "cut_off": best_entry.get("cut_off", False),
        "artifact_readiness": best_entry.get("artifact_readiness", empty_best_known(target_signature)["artifact_readiness"]),
        "onnx_sanity": best_entry.get("onnx_sanity", {}),
        "calibration": best_entry.get("calibration", {}),
        "mxq_handoff": best_entry.get("mxq_handoff", {}),
        "promotion_gates": best_entry.get("promotion_gates", {}),
        "source": best_entry.get("source"),
    }
    write_json(BEST_KNOWN_PATH, payload)
    return payload


def format_artifact_readiness(label: str, payload: dict[str, bool]) -> str:
    return (
        f"{label}[pth={payload.get('pth_ready', False)}, "
        f"onnx={payload.get('onnx_ready', False)}, "
        f"calibration={payload.get('calibration_ready', False)}, "
        f"mxq_handoff={payload.get('mxq_handoff_ready', False)}]"
    )


def result_status_label(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "n/a"
    if entry.get("cut_off"):
        return "cut_off"
    if entry.get("completed"):
        return "completed"
    return entry.get("status", "unknown")


def write_inbox_summary(
    best_known: dict[str, Any],
    latest_run: dict[str, Any] | None,
    budget_fully_used: bool,
    next_recommendation: str,
    investigation_finding: str,
) -> None:
    best_artifact = best_known.get("artifact_readiness") or {}
    latest_artifact = (latest_run or {}).get("artifact_readiness") or {}
    longer_follow_up = bool(latest_run and latest_run.get("cut_off") and (latest_run.get("delta_psnr") or 0) > 0)
    lines = [
        "# Lab 3 Autopilot Inbox Summary",
        "",
        f"- Current best comparable validation PSNR: {best_known.get('validation_psnr', 'n/a')}",
        f"- Current best comparable delta PSNR: {best_known.get('delta_psnr', 'n/a')}",
        f"- Whether the best run completed or was cut off: {result_status_label(best_known)}",
        f"- Whether the latest run completed or was cut off: {result_status_label(latest_run)}",
        (
            "- Artifact readiness for `.pth`, `.onnx`, calibration, and MXQ handoff: "
            f"{format_artifact_readiness('best', best_artifact)}; "
            f"{format_artifact_readiness('latest', latest_artifact)}"
        ),
        f"- Most important investigation finding: {investigation_finding}",
        f"- Next recommended mutation or architecture family: {next_recommendation}",
        f"- Whether the 4-hour budget was fully used: {str(budget_fully_used).lower()}",
        f"- Whether a longer follow-up run is justified: {str(longer_follow_up).lower()}",
    ]
    INBOX_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_id_for_candidate(candidate: CandidateConfig) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"autopilotmodal_{candidate.candidate_id}_{stamp}"


def extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.rfind("{")
        if start == -1:
            return {}
        try:
            return json.loads(stripped[start:])
        except json.JSONDecodeError:
            return {}


def launcher_failure_result(
    candidate: CandidateConfig,
    *,
    run_id: str,
    status: str,
    started_at: str,
    ended_at: str,
    wall_clock_minutes: float,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    result = RunResult(
        run_id=run_id,
        config_hash=stable_config_hash(candidate),
        source="controller",
        status=status,
        completed=False,
        cut_off=status == "cut_off",
        run_category=candidate.run_category,
        candidate={**candidate.duplicate_key(), "run_category": candidate.run_category},
        run_root="",
        summary_path=None,
        report_path=None,
        notebook_path=None,
        started_at=started_at,
        ended_at=ended_at,
        wall_clock_minutes=wall_clock_minutes,
        validation_psnr=None,
        delta_psnr=None,
        input_psnr=None,
        val_loss=None,
        artifact_readiness={
            "pth_ready": False,
            "onnx_ready": False,
            "calibration_ready": False,
            "mxq_handoff_ready": False,
        },
        comparison_signature=candidate.comparison_signature(),
        onnx_sanity={},
        calibration={},
        mxq_handoff={},
        promotion_gates={},
        notes=[
            "Launcher failed before a normalized summary could be recorded.",
            f"stdout: {stdout.strip()}" if stdout.strip() else "stdout: <empty>",
            f"stderr: {stderr.strip()}" if stderr.strip() else "stderr: <empty>",
        ],
        rerun_reason=candidate.rerun_reason,
    )
    return result.as_json()


def run_result_from_launcher_payload(
    candidate: CandidateConfig,
    payload: dict[str, Any],
    *,
    started_at: str,
    ended_at: str,
    wall_clock_minutes: float,
    rerun_reason: str | None,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    notes = list(payload.get("notes", []))
    if extra_notes:
        notes.extend(extra_notes)
    run_root = payload.get("run_root", "")
    fallback_run_id = Path(run_root).name if run_root else "unknown-run"
    result = RunResult(
        run_id=payload.get("run_id") or fallback_run_id,
        config_hash=stable_config_hash(candidate),
        source="controller",
        status=payload.get("status", "unknown"),
        completed=bool(payload.get("completed", False)),
        cut_off=bool(payload.get("cut_off", False)),
        run_category=candidate.run_category,
        candidate={**candidate.duplicate_key(), "run_category": candidate.run_category},
        run_root=run_root,
        summary_path=payload.get("summary_path"),
        report_path=payload.get("report_path"),
        notebook_path=payload.get("notebook_path"),
        started_at=started_at,
        ended_at=ended_at,
        wall_clock_minutes=wall_clock_minutes,
        validation_psnr=payload.get("validation_psnr"),
        delta_psnr=payload.get("delta_psnr"),
        input_psnr=payload.get("input_psnr"),
        val_loss=payload.get("val_loss"),
        artifact_readiness=payload.get(
            "artifact_readiness",
            {
                "pth_ready": False,
                "onnx_ready": False,
                "calibration_ready": False,
                "mxq_handoff_ready": False,
            },
        ),
        comparison_signature=payload.get("comparison_signature", candidate.comparison_signature()),
        onnx_sanity=payload.get("onnx_sanity", {}),
        calibration=payload.get("calibration", {}),
        mxq_handoff=payload.get("mxq_handoff", {}),
        promotion_gates=payload.get("promotion_gates", {}),
        notes=notes,
        rerun_reason=rerun_reason,
    )
    return result.as_json()


def launch_candidate(
    candidate: CandidateConfig,
    args: argparse.Namespace,
    deadline_monotonic: float,
    checkpoint_seconds: int,
    best_known: dict[str, Any],
    prior_best_val_psnr: float | None,
) -> dict[str, Any]:
    run_id = run_id_for_candidate(candidate)
    started_at = now_iso()
    started_monotonic = time.monotonic()
    command = [
        sys.executable,
        str(LAUNCH_SCRIPT),
        "--family",
        candidate.family,
        "--depth",
        str(candidate.depth),
        "--width",
        str(candidate.width),
        "--kernel-mix",
        candidate.kernel_mix,
        "--activation",
        candidate.activation,
        "--optimizer",
        candidate.optimizer,
        "--schedule",
        candidate.schedule,
        "--data-slice",
        candidate.data_slice,
        "--budget-minutes",
        str(candidate.budget_minutes),
        "--run-id",
        run_id,
        "--candidate-id",
        candidate.candidate_id,
        "--batch-size",
        str(candidate.batch_size),
        "--eval-size",
        str(candidate.eval_size),
        "--train-patch-size",
        str(candidate.train_patch_size),
        "--modal-gpu",
        candidate.modal_gpu,
        "--modal-data-volume",
        args.modal_data_volume,
        "--modal-runs-volume",
        args.modal_runs_volume,
        "--poll-interval-minutes",
        str(max(1, args.checkpoint_minutes)),
    ]
    if prior_best_val_psnr is not None:
        command.extend(["--prior-best-val-psnr", str(prior_best_val_psnr)])
    if args.sync_data:
        command.append("--sync-data")
    if args.force_data_sync:
        command.append("--force-data-sync")

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    next_checkpoint = started_monotonic + checkpoint_seconds
    cut_off = False

    while process.poll() is None:
        now_monotonic = time.monotonic()
        if now_monotonic >= deadline_monotonic:
            process.kill()
            process.wait()
            cut_off = True
            break
        if now_monotonic >= next_checkpoint:
            write_inbox_summary(
                best_known=best_known,
                latest_run=None,
                budget_fully_used=False,
                next_recommendation=f"Active: {candidate.candidate_id}",
                investigation_finding="Checkpoint recorded while the active Modal experiment was still running.",
            )
            next_checkpoint += checkpoint_seconds
        time.sleep(5)

    stdout, stderr = process.communicate()
    ended_at = now_iso()
    wall_clock_minutes = round((time.monotonic() - started_monotonic) / 60.0, 3)

    if cut_off:
        return launcher_failure_result(
            candidate,
            run_id=run_id,
            status="cut_off",
            started_at=started_at,
            ended_at=ended_at,
            wall_clock_minutes=wall_clock_minutes,
            stdout=stdout,
            stderr=stderr,
        )

    payload = extract_json_payload(stdout)
    if process.returncode != 0 and payload:
        extra_notes: list[str] = []
        if stderr.strip():
            extra_notes.append(f"launcher stderr: {stderr.strip()}")
        return run_result_from_launcher_payload(
            candidate,
            payload,
            started_at=started_at,
            ended_at=ended_at,
            wall_clock_minutes=wall_clock_minutes,
            rerun_reason=candidate.rerun_reason,
            extra_notes=extra_notes,
        )

    if process.returncode != 0 or not payload:
        return launcher_failure_result(
            candidate,
            run_id=run_id,
            status="launcher_failed",
            started_at=started_at,
            ended_at=ended_at,
            wall_clock_minutes=wall_clock_minutes,
            stdout=stdout,
            stderr=stderr,
        )

    extra_notes: list[str] = []
    if stderr.strip():
        extra_notes.append(f"launcher stderr: {stderr.strip()}")
    return run_result_from_launcher_payload(
        candidate,
        payload,
        started_at=started_at,
        ended_at=ended_at,
        wall_clock_minutes=wall_clock_minutes,
        rerun_reason=candidate.rerun_reason,
        extra_notes=extra_notes,
    )


def validate_canonical_pipeline(notebook_path: Path) -> None:
    command = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--notebook",
        str(notebook_path),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def select_next_candidate(
    ledger_entries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[CandidateConfig | None, str]:
    target_signature = target_signature_from_args(args)
    comparable = comparable_entries(ledger_entries, target_signature)
    seen_hashes = {entry.get("config_hash") for entry in comparable}
    seen_candidate_ids = {
        entry.get("candidate", {}).get("candidate_id")
        for entry in comparable
        if entry.get("candidate", {}).get("candidate_id")
    }
    real_metric_entries = [entry for entry in comparable if entry.get("validation_psnr") is not None]

    if args.force_candidate:
        forced = candidate_from_registry(args.force_candidate, args, rerun_reason=args.rerun_reason or None)
        if stable_config_hash(forced) in seen_hashes and not args.rerun_reason:
            return None, (
                f"forced candidate {args.force_candidate} already exists for the target signature; "
                "record a rerun_reason to rerun it"
            )
        return forced, f"forced candidate requested: {args.force_candidate}"

    baseline = candidate_from_registry("wide_residual_nobn_v1", args, run_category="benchmark", rerun_reason=args.rerun_reason or None)
    baseline_hash = stable_config_hash(baseline)
    if baseline_hash not in seen_hashes:
        return baseline, "no comparable baseline exists for the target signature"

    get_candidate_spec, list_candidate_ids = import_pipeline_helpers()
    best_entry = max(real_metric_entries, key=lambda entry: entry.get("validation_psnr", float("-inf"))) if real_metric_entries else None
    best_candidate_id = best_entry.get("candidate", {}).get("candidate_id") if best_entry else "wide_residual_nobn_v1"

    bounded_ids = list_candidate_ids(include_extreme=False)
    unseen_bounded = []
    for candidate_id in bounded_ids:
        candidate = candidate_from_registry(candidate_id, args, rerun_reason=args.rerun_reason or None)
        if stable_config_hash(candidate) not in seen_hashes:
            unseen_bounded.append(candidate_id)

    if unseen_bounded:
        neighbors = sorted(
            unseen_bounded,
            key=lambda candidate_id: (
                candidate_distance(candidate_id, best_candidate_id),
                bounded_ids.index(candidate_id),
            ),
        )
        chosen = neighbors[0]
        return candidate_from_registry(chosen, args, rerun_reason=args.rerun_reason or None), (
            f"selected unseen bounded candidate {chosen} nearest to current best comparable {best_candidate_id}"
        )

    if args.rerun_reason:
        rerun_target = best_candidate_id or "wide_residual_nobn_v1"
        return candidate_from_registry(rerun_target, args, rerun_reason=args.rerun_reason), (
            f"rerunning {rerun_target} because rerun_reason was provided"
        )

    if args.allow_extreme:
        extreme_ids = [
            candidate_id
            for candidate_id in list_candidate_ids(include_extreme=True)
            if candidate_id not in bounded_ids
        ]
        unseen_extreme = []
        for candidate_id in extreme_ids:
            candidate = candidate_from_registry(candidate_id, args)
            if stable_config_hash(candidate) not in seen_hashes:
                unseen_extreme.append(candidate_id)
        if unseen_extreme:
            chosen = sorted(unseen_extreme, key=lambda candidate_id: candidate_distance(candidate_id, best_candidate_id))[0]
            return candidate_from_registry(chosen, args), (
                f"bounded candidates exhausted; escalating to unseen investigation candidate {chosen}"
            )

    return None, "no unseen comparable candidate remains; add a rerun reason or register a new candidate"


def recommend_next_mutation(ledger_entries: list[dict[str, Any]], args: argparse.Namespace) -> str:
    candidate, reason = select_next_candidate(ledger_entries, argparse.Namespace(**{**vars(args), "rerun_reason": "", "force_candidate": ""}))
    if candidate is None:
        return reason
    return f"{candidate.candidate_id} ({reason})"


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.canonical_notebook).expanduser().resolve()
    total_budget_seconds = int(args.total_budget_hours * 3600)
    checkpoint_seconds = max(60, args.checkpoint_minutes * 60)
    target_signature = target_signature_from_args(args)

    ensure_reports_dir()
    validate_canonical_pipeline(notebook_path)

    ledger_entries = bootstrap_or_refresh_ledger()
    best_known = rebuild_best_known(ledger_entries, target_signature)
    latest_run: dict[str, Any] | None = None

    started_monotonic = time.monotonic()
    deadline_monotonic = started_monotonic + total_budget_seconds

    for _slot_index in range(max(1, args.max_runs)):
        candidate, selection_reason = select_next_candidate(ledger_entries, args)
        if candidate is None:
            break

        remaining_seconds = max(0, int(deadline_monotonic - time.monotonic()))
        if remaining_seconds <= 0:
            break

        candidate_budget = min(candidate.budget_minutes, max(1, remaining_seconds // 60))
        runnable_candidate = CandidateConfig(
            candidate_id=candidate.candidate_id,
            family=candidate.family,
            depth=candidate.depth,
            width=candidate.width,
            kernel_mix=candidate.kernel_mix,
            activation=candidate.activation,
            optimizer=candidate.optimizer,
            schedule=candidate.schedule,
            data_slice=candidate.data_slice,
            budget_minutes=candidate_budget,
            batch_size=candidate.batch_size,
            eval_size=candidate.eval_size,
            train_patch_size=candidate.train_patch_size,
            modal_gpu=candidate.modal_gpu,
            run_category=candidate.run_category,
            rerun_reason=candidate.rerun_reason,
        )

        prior_best = best_val_psnr_for_signature(ledger_entries, runnable_candidate.comparison_signature())
        latest_run = launch_candidate(
            candidate=runnable_candidate,
            args=args,
            deadline_monotonic=deadline_monotonic,
            checkpoint_seconds=checkpoint_seconds,
            best_known=best_known,
            prior_best_val_psnr=prior_best,
        )
        latest_run.setdefault("notes", []).insert(0, f"selection_reason: {selection_reason}")
        append_jsonl(LEDGER_PATH, latest_run)
        ledger_entries.append(latest_run)
        best_known = rebuild_best_known(ledger_entries, target_signature)
        write_inbox_summary(
            best_known=best_known,
            latest_run=latest_run,
            budget_fully_used=False,
            next_recommendation=recommend_next_mutation(ledger_entries, args),
            investigation_finding=(
                "Autopilot is using real synced Modal summaries for ranking, artifact readiness, "
                "and duplicate avoidance."
            ),
        )

    budget_fully_used = time.monotonic() >= deadline_monotonic
    write_inbox_summary(
        best_known=best_known,
        latest_run=latest_run,
        budget_fully_used=budget_fully_used,
        next_recommendation=recommend_next_mutation(ledger_entries, args),
        investigation_finding=(
            "Controller now parses real validation metrics, ONNX sanity, calibration outputs, "
            "and MXQ handoff readiness from synced completed runs. TODO: add smarter candidate mutations "
            "beyond nearest-neighbor selection if search needs to widen."
        ),
    )


if __name__ == "__main__":
    main()
