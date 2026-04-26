from __future__ import annotations

import copy
import json
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from lab3_pipeline_lib import PipelineConfig, pipeline_config_from_json


APP_NAME = "lab3-modal-pipeline"
FUNCTION_NAME = "run_lab3_pipeline"
DEFAULT_GPU = "L40S"
DEFAULT_TIMEOUT_MINUTES = 120
DEFAULT_POLL_INTERVAL_MINUTES = 30
DEFAULT_DATA_VOLUME_NAME = "lab3-data"
DEFAULT_RUNS_VOLUME_NAME = "lab3-runs"

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_ROOT = PurePosixPath("/root/project")
REMOTE_DATA_VOLUME_ROOT = PurePosixPath("/mnt/lab3-data")
REMOTE_RUNS_VOLUME_ROOT = PurePosixPath("/mnt/lab3-runs")
REMOTE_DATA_ROOT = REMOTE_DATA_VOLUME_ROOT / "Data"


def _modal_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install(
            "torch==2.10.0",
            "onnx==1.20.1",
            "onnxruntime==1.24.1",
            "numpy==2.4.2",
            "pillow==12.1.1",
        )
        .add_local_dir(
            LOCAL_PROJECT_ROOT / "tools",
            remote_path=str(REMOTE_PROJECT_ROOT / "tools"),
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_file(
            LOCAL_PROJECT_ROOT / "lab3_step2_onnx_to_mxq.py",
            remote_path=str(REMOTE_PROJECT_ROOT / "lab3_step2_onnx_to_mxq.py"),
        )
    )


def _run_modal_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True, cwd=str(LOCAL_PROJECT_ROOT))


def ensure_volume(name: str) -> None:
    volume = modal.Volume.from_name(name, create_if_missing=True)
    volume.hydrate()


def volume_path_exists(volume_name: str, remote_path: str) -> bool:
    completed = _run_modal_command(["modal", "volume", "ls", volume_name, remote_path])
    return completed.returncode == 0


def sync_data_volume(local_data_root: Path, volume_name: str, force: bool = False) -> dict[str, Any]:
    ensure_volume(volume_name)
    if not force and volume_path_exists(volume_name, "/Data"):
        return {
            "status": "reused",
            "local_data_root": str(local_data_root),
            "volume_name": volume_name,
            "remote_path": "/Data",
        }

    args = ["modal", "volume", "put", volume_name, str(local_data_root), "/Data", "--force"]
    completed = _run_modal_command(args)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to sync data volume {volume_name}: {completed.stderr.strip() or completed.stdout.strip()}")
    return {
        "status": "synced",
        "local_data_root": str(local_data_root),
        "volume_name": volume_name,
        "remote_path": "/Data",
    }


def sync_run_from_volume(run_day: str, run_name: str, volume_name: str, local_project_root: Path) -> Path:
    ensure_volume(volume_name)
    local_day_root = local_project_root / "runs" / run_day
    local_day_root.mkdir(parents=True, exist_ok=True)
    remote_path = f"/runs/{run_day}/{run_name}"
    completed = _run_modal_command(
        ["modal", "volume", "get", volume_name, remote_path, str(local_day_root), "--force"]
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to sync remote run {remote_path}: {completed.stderr.strip() or completed.stdout.strip()}")
    return local_day_root / run_name


def _replace_prefix(value: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        if value.startswith(old):
            return new + value[len(old) :]
    return value


def _localize_runtime_path(value: str, local_run_root: Path, local_data_root: Path) -> str:
    run_marker = f"/runs/{local_run_root.parent.name}/{local_run_root.name}"
    if run_marker in value:
        _, suffix = value.split(run_marker, 1)
        return str(local_run_root) + suffix
    data_marker = "/Data/"
    if data_marker in value:
        _, suffix = value.split(data_marker, 1)
        return str(local_data_root / suffix)
    if value.endswith("/lab3_step2_onnx_to_mxq.py"):
        return str(LOCAL_PROJECT_ROOT / "lab3_step2_onnx_to_mxq.py")
    return value


def _normalize_payload(payload: Any, replacements: dict[str, str], local_run_root: Path, local_data_root: Path) -> Any:
    if isinstance(payload, dict):
        return {key: _normalize_payload(value, replacements, local_run_root, local_data_root) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalize_payload(value, replacements, local_run_root, local_data_root) for value in payload]
    if isinstance(payload, str):
        return _localize_runtime_path(_replace_prefix(payload, replacements), local_run_root, local_data_root)
    return payload


def path_replacements(local_project_root: Path, local_data_root: Path) -> dict[str, str]:
    return {
        str(REMOTE_RUNS_VOLUME_ROOT): str(local_project_root),
        str(REMOTE_PROJECT_ROOT): str(local_project_root),
        str(REMOTE_DATA_VOLUME_ROOT): str(local_project_root),
        str(REMOTE_DATA_ROOT): str(local_data_root),
    }


def normalize_synced_run(local_run_root: Path, local_project_root: Path, local_data_root: Path) -> None:
    replacements = path_replacements(local_project_root, local_data_root)
    for name in ["run_config.json", "summary.json", "report.json", "latest_status.json"]:
        path = local_run_root / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        normalized = _normalize_payload(payload, replacements, local_run_root, local_data_root)
        path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_artifact_readiness(local_run_root: Path) -> dict[str, bool]:
    return {
        "pth_ready": (local_run_root / "checkpoints" / "best.pt").exists(),
        "onnx_ready": (local_run_root / "exports" / "best.onnx").exists(),
        "calibration_ready": (local_run_root / "exports" / "calibration" / "manifest.json").exists(),
        "mxq_handoff_ready": (local_run_root / "report.json").exists() or (local_run_root / "summary.json").exists(),
    }


def _load_last_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    last_line = ""
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        return {}
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return {}


def build_follow_up_recommendation(summary: dict[str, Any], was_cut_off: bool) -> dict[str, Any]:
    evaluation = summary.get("evaluation", {})
    latest_metrics = summary.get("training", {}).get("latest_metrics") or {}
    delta_psnr = evaluation.get("delta_psnr", latest_metrics.get("delta_psnr"))
    val_psnr = evaluation.get("val_psnr", latest_metrics.get("val_psnr"))
    if not was_cut_off:
        return {
            "fuller_run_justified": False,
            "reason": "Run completed within the current time budget.",
            "observed_val_psnr": val_psnr,
            "observed_delta_psnr": delta_psnr,
        }
    justified = bool(delta_psnr is not None and delta_psnr > 0.0)
    reason = "Positive validation delta at cutoff suggests more training may pay off." if justified else "No positive validation delta before cutoff."
    return {
        "fuller_run_justified": justified,
        "reason": reason,
        "observed_val_psnr": val_psnr,
        "observed_delta_psnr": delta_psnr,
    }


def _partial_summary(local_run_root: Path, cfg: PipelineConfig, heartbeats: list[dict[str, Any]], was_cut_off: bool) -> dict[str, Any]:
    config_payload = {}
    if (local_run_root / "run_config.json").exists():
        config_payload = json.loads((local_run_root / "run_config.json").read_text(encoding="utf-8"))
    latest_status = {}
    if (local_run_root / "latest_status.json").exists():
        latest_status = json.loads((local_run_root / "latest_status.json").read_text(encoding="utf-8"))
    latest_metrics = _load_last_metrics(local_run_root / "metrics.jsonl")
    summary = {
        "backend": "modal",
        "run_day": cfg.resolved_started_day(),
        "run_root": str(local_run_root),
        "summary_path": str(local_run_root / "summary.json"),
        "report_path": str(local_run_root / "report.json"),
        "notebook_path": str(local_run_root / "notebooks" / f"lab3_{cfg.candidate().notebook_slug}_autopilot.ipynb"),
        "candidate": cfg.candidate().as_json(),
        "training": {
            "latest_metrics": latest_metrics,
        },
        "evaluation": {
            key: latest_metrics.get(key)
            for key in ["val_psnr", "delta_psnr", "input_psnr", "val_loss"]
            if key in latest_metrics
        },
        "gates": {"promotion_pass": False, "screening_pass": False},
        "status": latest_status,
        "config": config_payload,
    }
    summary["artifact_readiness"] = _build_artifact_readiness(local_run_root)
    summary["follow_up_recommendation"] = build_follow_up_recommendation(summary, was_cut_off)
    summary["execution"] = {
        "backend": "modal",
        "final_status": "cut_off" if was_cut_off else "incomplete",
        "heartbeats": heartbeats,
        "synced_local_run_root": str(local_run_root),
    }
    return summary


def hydrate_synced_summary(
    local_run_root: Path,
    cfg: PipelineConfig,
    local_data_root: Path,
    fallback_summary: dict[str, Any] | None,
    heartbeats: list[dict[str, Any]],
    was_cut_off: bool,
) -> dict[str, Any]:
    normalize_synced_run(local_run_root, LOCAL_PROJECT_ROOT, local_data_root)
    report_path = local_run_root / "report.json"
    if report_path.exists():
        summary = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        summary_path = local_run_root / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            summary = payload.get("summary", payload)
        elif fallback_summary:
            summary = _normalize_payload(
                fallback_summary,
                path_replacements(LOCAL_PROJECT_ROOT, local_data_root),
                local_run_root,
                local_data_root,
            )
        else:
            summary = _partial_summary(local_run_root, cfg, heartbeats, was_cut_off)
    summary["artifact_readiness"] = _build_artifact_readiness(local_run_root)
    summary["follow_up_recommendation"] = build_follow_up_recommendation(summary, was_cut_off)
    execution = summary.setdefault("execution", {})
    execution.update(
        {
            "backend": "modal",
            "final_status": "cut_off" if was_cut_off else execution.get("final_status", "completed"),
            "heartbeats": heartbeats,
            "modal_app_name": APP_NAME,
            "modal_function_name": FUNCTION_NAME,
            "modal_gpu": cfg.modal_gpu,
            "modal_timeout_minutes": cfg.modal_timeout_minutes,
            "modal_data_volume": cfg.modal_data_volume,
            "modal_runs_volume": cfg.modal_runs_volume,
            "synced_local_run_root": str(local_run_root),
        }
    )
    summary["synced_local_run_root"] = str(local_run_root)
    return summary


def _remote_config(
    cfg: PipelineConfig,
    *,
    data_volume_name: str,
    runs_volume_name: str,
    gpu: str,
    timeout_minutes: int,
) -> PipelineConfig:
    remote_cfg = copy.deepcopy(cfg)
    remote_cfg.project_root = Path(str(REMOTE_PROJECT_ROOT))
    remote_cfg.data_root = Path(str(REMOTE_DATA_ROOT))
    remote_cfg.artifact_root = Path(str(REMOTE_RUNS_VOLUME_ROOT))
    remote_cfg.backend = "modal"
    remote_cfg.started_day = cfg.resolved_started_day()
    remote_cfg.modal_app_name = APP_NAME
    remote_cfg.modal_function_name = FUNCTION_NAME
    remote_cfg.modal_gpu = gpu
    remote_cfg.modal_timeout_minutes = timeout_minutes
    remote_cfg.modal_data_volume = data_volume_name
    remote_cfg.modal_runs_volume = runs_volume_name
    return remote_cfg


def make_modal_entrypoint(
    *,
    gpu: str,
    timeout_minutes: int,
    data_volume_name: str,
    runs_volume_name: str,
) -> tuple[modal.App, Any]:
    data_volume = modal.Volume.from_name(data_volume_name, create_if_missing=True)
    runs_volume = modal.Volume.from_name(runs_volume_name, create_if_missing=True)
    app = modal.App(APP_NAME)
    image = _modal_image()

    @app.function(
        image=image,
        gpu=gpu,
        serialized=True,
        timeout=timeout_minutes * 60,
        volumes={
            str(REMOTE_DATA_VOLUME_ROOT): data_volume,
            str(REMOTE_RUNS_VOLUME_ROOT): runs_volume,
        },
        name=FUNCTION_NAME,
    )
    def run_lab3_pipeline(config_payload: dict[str, Any], prior_best_val_psnr: float | None = None) -> dict[str, Any]:
        import sys

        sys.path.insert(0, str(REMOTE_PROJECT_ROOT / "tools"))
        from lab3_pipeline_lib import pipeline_config_from_json, run_pipeline

        cfg = pipeline_config_from_json(config_payload)
        summary = run_pipeline(cfg, prior_best_val_psnr=prior_best_val_psnr)
        runs_volume.commit()
        return summary

    return app, run_lab3_pipeline


def execute_modal_pipeline(
    cfg: PipelineConfig,
    *,
    prior_best_val_psnr: float | None,
    sync_data: bool,
    force_data_sync: bool,
    poll_interval_minutes: int = DEFAULT_POLL_INTERVAL_MINUTES,
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
    gpu: str = DEFAULT_GPU,
    data_volume_name: str = DEFAULT_DATA_VOLUME_NAME,
    runs_volume_name: str = DEFAULT_RUNS_VOLUME_NAME,
) -> dict[str, Any]:
    if sync_data or force_data_sync or not volume_path_exists(data_volume_name, "/Data"):
        data_sync = sync_data_volume(cfg.data_root, data_volume_name, force=force_data_sync)
    else:
        ensure_volume(data_volume_name)
        ensure_volume(runs_volume_name)
        data_sync = {
            "status": "skipped",
            "local_data_root": str(cfg.data_root),
            "volume_name": data_volume_name,
            "remote_path": "/Data",
        }

    remote_cfg = _remote_config(
        cfg,
        data_volume_name=data_volume_name,
        runs_volume_name=runs_volume_name,
        gpu=gpu,
        timeout_minutes=timeout_minutes,
    )
    remote_payload = remote_cfg.as_json()
    app, remote_fn = make_modal_entrypoint(
        gpu=gpu,
        timeout_minutes=timeout_minutes,
        data_volume_name=data_volume_name,
        runs_volume_name=runs_volume_name,
    )

    heartbeat_seconds = max(60, poll_interval_minutes * 60)
    deadline = time.time() + timeout_minutes * 60
    heartbeats: list[dict[str, Any]] = []
    fallback_summary: dict[str, Any] | None = None
    was_cut_off = False

    with app.run():
        call = remote_fn.spawn(remote_payload, prior_best_val_psnr=prior_best_val_psnr)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                was_cut_off = True
                call.cancel()
                break
            try:
                fallback_summary = call.get(timeout=min(heartbeat_seconds, remaining))
                break
            except TimeoutError:
                heartbeats.append(
                    {
                        "elapsed_minutes": round((time.time() - (deadline - timeout_minutes * 60)) / 60.0, 2),
                        "status": "still_running",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    }
                )
            except Exception:
                raise

    local_run_root = sync_run_from_volume(remote_cfg.started_day, remote_cfg.run_name, runs_volume_name, LOCAL_PROJECT_ROOT)
    summary = hydrate_synced_summary(local_run_root, remote_cfg, cfg.data_root, fallback_summary, heartbeats, was_cut_off)
    summary.setdefault("execution", {})["data_volume_sync"] = data_sync
    if fallback_summary and not was_cut_off:
        summary.setdefault("execution", {})["remote_completion"] = "returned"
    return summary
