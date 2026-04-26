from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from restormer_teacher.logging_utils import atomic_write_json, atomic_write_jsonl, load_history, load_jsonl


@dataclass(frozen=True)
class ReconciledRunState:
    history: dict[str, Any] | None
    metrics_records: list[dict[str, Any]]
    latest_status: dict[str, Any] | None


def _epoch_sequence(records: list[dict[str, Any]]) -> list[int]:
    return [int(rec["epoch"]) for rec in records if rec.get("kind") == "epoch"]


def _validate_epoch_sequence(name: str, epochs: list[int]) -> None:
    if any(cur <= prev for prev, cur in zip(epochs, epochs[1:])):
        raise RuntimeError(f"{name} must be strictly increasing; got {epochs}")


def _validate_run_meta(existing: dict[str, Any] | None, expected: dict[str, Any]) -> None:
    if not existing:
        return
    existing_meta = dict(existing.get("meta", {}))
    for key in ("teacher_model_version", "config_fingerprint", "run_id"):
        if key in existing_meta and key in expected and existing_meta[key] != expected[key]:
            raise RuntimeError(
                f"Existing run metadata mismatch for {key}: {existing_meta[key]!r} != {expected[key]!r}"
            )


def _rewrite_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, records)


def _truncate_metrics(records: list[dict[str, Any]], keep_max_epoch: int) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for rec in records:
        epoch = rec.get("epoch")
        if epoch is None:
            trimmed.append(rec)
            continue
        if int(epoch) <= keep_max_epoch:
            trimmed.append(rec)
    return trimmed


def _trim_history(history: dict[str, Any] | None, keep_max_epoch: int) -> dict[str, Any] | None:
    if history is None:
        return None
    epochs = [rec for rec in history.get("epochs", []) if int(rec["epoch"]) <= keep_max_epoch]
    return {"meta": dict(history.get("meta", {})), "epochs": epochs}


def reconcile_run_state(
    *,
    run_root: Path,
    history_meta: dict[str, Any],
    resume_epoch: int | None,
    allow_legacy_checkpoint: bool,
) -> ReconciledRunState:
    history_path = run_root / "history.json"
    metrics_path = run_root / "metrics.jsonl"
    latest_status_path = run_root / "latest_status.json"

    history = load_history(history_path)
    metrics_records = load_jsonl(metrics_path)
    latest_status = None
    if latest_status_path.exists():
        latest_status = json.loads(latest_status_path.read_text(encoding="utf-8"))

    _validate_run_meta(history, history_meta)

    history_epochs = [int(rec["epoch"]) for rec in history.get("epochs", [])] if history else []
    metric_epochs = _epoch_sequence(metrics_records)
    _validate_epoch_sequence("history.json epochs", history_epochs)
    _validate_epoch_sequence("metrics.jsonl epoch rows", metric_epochs)

    if history_epochs != metric_epochs:
        raise RuntimeError(
            "history.json and metrics.jsonl epoch rows diverged: "
            f"{history_epochs} vs {metric_epochs}"
        )
    if latest_status is not None and metric_epochs:
        latest_epoch = int(latest_status.get("epoch", -1))
        if latest_epoch != metric_epochs[-1]:
            raise RuntimeError(
                f"latest_status.json epoch {latest_epoch} does not match metrics tail {metric_epochs[-1]}"
            )

    if resume_epoch is None:
        if metric_epochs:
            raise RuntimeError(
                f"Run directory {run_root} already contains epoch records; resume explicitly or use a new run_name."
            )
        return ReconciledRunState(history=history, metrics_records=metrics_records, latest_status=latest_status)

    keep_max_epoch = int(resume_epoch)
    if metric_epochs and metric_epochs[-1] < keep_max_epoch and not allow_legacy_checkpoint:
        raise RuntimeError(
            f"Checkpoint epoch {keep_max_epoch} is ahead of logged epoch {metric_epochs[-1]} in {run_root}"
        )

    trimmed_history = _trim_history(history, keep_max_epoch)
    trimmed_metrics = _truncate_metrics(metrics_records, keep_max_epoch)
    history_changed = trimmed_history != history
    metrics_changed = trimmed_metrics != metrics_records

    if latest_status is not None:
        latest_epoch = int(latest_status.get("epoch", -1))
        if latest_epoch > keep_max_epoch:
            latest_status = None
            latest_status_path.unlink()

    if history_changed:
        if trimmed_history is None:
            history_path.unlink(missing_ok=True)
        else:
            atomic_write_json(history_path, trimmed_history)
        history = trimmed_history
    if metrics_changed:
        _rewrite_metrics(metrics_path, trimmed_metrics)
        metrics_records = trimmed_metrics

    history_epochs = [int(rec["epoch"]) for rec in history.get("epochs", [])] if history else []
    metric_epochs = _epoch_sequence(metrics_records)
    if history_epochs != metric_epochs:
        raise RuntimeError(
            "Run-state reconciliation failed to realign history.json and metrics.jsonl: "
            f"{history_epochs} vs {metric_epochs}"
        )

    return ReconciledRunState(history=history, metrics_records=metrics_records, latest_status=latest_status)
