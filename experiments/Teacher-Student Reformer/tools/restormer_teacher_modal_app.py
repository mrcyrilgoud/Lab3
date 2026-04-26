from __future__ import annotations

import copy
from dataclasses import replace
import json
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import modal

if str((Path(__file__).resolve().parents[1])) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if TYPE_CHECKING:
    from restormer_teacher.config import TeacherResolvedConfig

APP_NAME = "restormer-teacher"
FUNCTION_NAME = "run_restormer_teacher_train"
DEFAULT_GPU = "L40S"
DEFAULT_TIMEOUT_MINUTES = 1440
DEFAULT_POLL_INTERVAL_MINUTES = 5
DEFAULT_DATA_VOLUME_NAME = "lab3-data"
DEFAULT_RUNS_VOLUME_NAME = "lab3-runs"

# This file: Teacher-Student Reformer/tools/restormer_teacher_modal_app.py
_LOCAL_TOOLS_DIR = Path(__file__).resolve().parent
_LOCAL_PKG_ROOT = _LOCAL_TOOLS_DIR.parent
_LOCAL_LAB3_ROOT = _LOCAL_PKG_ROOT.parent

REMOTE_PKG_ROOT = PurePosixPath("/root/tsr_pkg")
REMOTE_LAB3_ROOT = PurePosixPath("/root/lab3")
REMOTE_DATA_VOLUME_ROOT = PurePosixPath("/mnt/lab3-data")
REMOTE_RUNS_VOLUME_ROOT = PurePosixPath("/mnt/lab3-runs")
REMOTE_DATA_ROOT = REMOTE_DATA_VOLUME_ROOT / "Data"


def _run_modal_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True, cwd=str(_LOCAL_LAB3_ROOT))


def ensure_volume(name: str) -> None:
    volume = modal.Volume.from_name(name, create_if_missing=True)
    volume.hydrate()


def volume_path_exists(volume_name: str, remote_path: str) -> bool:
    completed = _run_modal_command(["modal", "volume", "ls", volume_name, remote_path])
    return completed.returncode == 0


def sync_data_volume(local_data_root: Path, volume_name: str, force: bool = False) -> dict[str, Any]:
    ensure_volume(volume_name)
    if not force and volume_path_exists(volume_name, "/Data"):
        return {"status": "reused", "local_data_root": str(local_data_root), "volume_name": volume_name}
    args = ["modal", "volume", "put", volume_name, str(local_data_root), "/Data", "--force"]
    completed = _run_modal_command(args)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to sync data volume {volume_name}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return {"status": "synced", "local_data_root": str(local_data_root), "volume_name": volume_name}


def sync_run_from_volume(run_day: str, run_name: str, volume_name: str, local_project_root: Path) -> Path:
    ensure_volume(volume_name)
    local_day_root = local_project_root / "runs" / run_day
    local_day_root.mkdir(parents=True, exist_ok=True)
    remote_path = f"/runs/{run_day}/{run_name}"
    completed = _run_modal_command(
        ["modal", "volume", "get", volume_name, remote_path, str(local_day_root), "--force"]
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to sync remote run {remote_path}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return local_day_root / run_name


def _remote_run_path(run_day: str, run_name: str) -> str:
    return f"/runs/{run_day}/{run_name}"


def remote_run_exists(run_day: str, run_name: str, volume_name: str) -> bool:
    return volume_path_exists(volume_name, _remote_run_path(run_day, run_name))


def normalize_synced_teacher_run(local_run_root: Path, local_project_root: Path, local_data_root: Path) -> None:
    root_tools = local_project_root / "tools"
    if str(root_tools) not in sys.path:
        sys.path.insert(0, str(root_tools))
    try:
        from lab3_modal_app import normalize_synced_run
    except Exception:
        return
    normalize_synced_run(local_run_root, local_project_root, local_data_root)


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _teacher_artifact_readiness(local_run_root: Path) -> dict[str, bool]:
    return {
        "best_checkpoint_ready": (local_run_root / "checkpoints" / "best_ema.pth").exists(),
        "latest_checkpoint_ready": (local_run_root / "checkpoints" / "latest.pth").exists(),
        "history_ready": (local_run_root / "history.json").exists(),
        "metrics_ready": (local_run_root / "metrics.jsonl").exists(),
    }


def _existing_run_fingerprint(local_lab3_root: Path, started_day: str, run_name: str) -> str:
    history_path = local_lab3_root / "runs" / started_day / run_name / "history.json"
    if not history_path.exists():
        return ""
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(history.get("meta", {}).get("config_fingerprint", ""))


def hydrate_synced_teacher_summary(
    *,
    local_run_root: Path,
    cfg: "TeacherResolvedConfig",
    local_data_root: Path,
    fallback_summary: dict[str, Any] | None,
    heartbeats: list[dict[str, Any]],
    was_cut_off: bool,
) -> dict[str, Any]:
    normalize_synced_teacher_run(local_run_root, _LOCAL_LAB3_ROOT, local_data_root)
    summary = dict(fallback_summary or {})
    latest_status = _load_json_if_exists(local_run_root / "latest_status.json")
    run_config = _load_json_if_exists(local_run_root / "run_config.json")
    history = _load_json_if_exists(local_run_root / "history.json")
    if not summary:
        summary = {
            "run_root": str(local_run_root),
            "teacher_model_version": cfg.teacher_model_version,
            "config_fingerprint": cfg.config_fingerprint,
        }
    summary["local_run_root"] = str(local_run_root)
    summary["latest_status"] = latest_status
    summary["run_config"] = run_config
    if history:
        summary["history_meta"] = history.get("meta", {})
        summary["epochs_logged"] = len(history.get("epochs", []))
    summary["artifact_readiness"] = _teacher_artifact_readiness(local_run_root)
    summary["execution"] = {
        "backend": "modal",
        "heartbeats": heartbeats,
        "was_cut_off": was_cut_off,
        "final_status": "cut_off" if was_cut_off else "completed",
        "modal_app_name": APP_NAME,
        "modal_function_name": FUNCTION_NAME,
        "synced_local_run_root": str(local_run_root),
    }
    return summary


def finalize_modal_result(
    *,
    cfg: "TeacherResolvedConfig",
    local_lab3_root: Path,
    local_data_root: Path,
    runs_volume_name: str,
    fallback_summary: dict[str, Any] | None,
    heartbeats: list[dict[str, Any]],
    was_cut_off: bool,
) -> dict[str, Any]:
    run_day = cfg.training.started_day
    run_name = cfg.training.run_name
    if not remote_run_exists(run_day, run_name, runs_volume_name):
        return {
            "run_root": str(local_lab3_root / "runs" / run_day / run_name),
            "teacher_model_version": cfg.teacher_model_version,
            "config_fingerprint": cfg.config_fingerprint,
            "execution": {
                "backend": "modal",
                "heartbeats": heartbeats,
                "was_cut_off": was_cut_off,
                "final_status": "cut_off" if was_cut_off else "missing_remote_run",
                "modal_app_name": APP_NAME,
                "modal_function_name": FUNCTION_NAME,
                "remote_run_path": _remote_run_path(run_day, run_name),
            },
            "artifact_readiness": {
                "best_checkpoint_ready": False,
                "latest_checkpoint_ready": False,
                "history_ready": False,
                "metrics_ready": False,
            },
        }
    local_run_root = sync_run_from_volume(run_day, run_name, runs_volume_name, local_lab3_root)
    return hydrate_synced_teacher_summary(
        local_run_root=local_run_root,
        cfg=cfg,
        local_data_root=local_data_root,
        fallback_summary=fallback_summary,
        heartbeats=heartbeats,
        was_cut_off=was_cut_off,
    )


def _modal_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install(
            "torch",
            "numpy",
            "pillow",
            "pyyaml",
        )
        .add_local_dir(
            str(_LOCAL_PKG_ROOT),
            remote_path=str(REMOTE_PKG_ROOT),
            ignore=["__pycache__", "*.pyc", ".ipynb_checkpoints"],
        )
        .add_local_dir(
            str(_LOCAL_LAB3_ROOT / "tools"),
            remote_path=str(REMOTE_LAB3_ROOT / "tools"),
            ignore=["__pycache__", "*.pyc"],
        )
    )


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
    def run_restormer_teacher_train(payload: dict[str, Any]) -> dict[str, Any]:
        sys.path.insert(0, str(REMOTE_PKG_ROOT))
        sys.path.insert(0, str(REMOTE_LAB3_ROOT / "tools"))
        from pathlib import Path as _Path

        from restormer_teacher.config import teacher_config_from_json
        from restormer_teacher.train import run_training

        cfg = teacher_config_from_json(copy.deepcopy(payload["config"]))
        cfg = replace(
            cfg,
            project_root=_Path(str(REMOTE_LAB3_ROOT)),
        ).with_training_updates(
            data_root=_Path(str(REMOTE_DATA_ROOT)),
            artifact_root=_Path(str(REMOTE_RUNS_VOLUME_ROOT / "runs")),
            modal_app=APP_NAME,
        )
        resume_path = None
        resume_remote = payload.get("resume_remote_path")
        if resume_remote:
            candidate = _Path(str(resume_remote))
            if candidate.is_file():
                resume_path = candidate

        summary = run_training(cfg, resume_path=resume_path)
        runs_volume.commit()
        return summary

    return app, run_restormer_teacher_train


def execute_restormer_teacher_modal(
    raw_cfg: dict[str, Any],
    *,
    local_lab3_root: Path,
    local_data_root: Path,
    run_name: str,
    started_day: str,
    resume_remote_path: str = "",
    smoke_test: bool = False,
    detach: bool = False,
    sync_data: bool = False,
    force_data_sync: bool = False,
    gpu: str = DEFAULT_GPU,
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
    poll_interval_minutes: int = DEFAULT_POLL_INTERVAL_MINUTES,
    data_volume_name: str = DEFAULT_DATA_VOLUME_NAME,
    runs_volume_name: str = DEFAULT_RUNS_VOLUME_NAME,
) -> dict[str, Any]:
    from restormer_teacher.config import resolve_teacher_config

    ensure_volume(runs_volume_name)
    if sync_data or force_data_sync or not volume_path_exists(data_volume_name, "/Data"):
        data_sync = sync_data_volume(local_data_root, data_volume_name, force=force_data_sync)
    else:
        ensure_volume(data_volume_name)
        data_sync = {"status": "skipped", "local_data_root": str(local_data_root), "volume_name": data_volume_name}

    raw = copy.deepcopy(raw_cfg)
    tr = raw.setdefault("training", {})
    tr["run_name"] = run_name
    tr["started_day"] = started_day
    if smoke_test:
        tr["smoke_test_override"] = True
    cfg = resolve_teacher_config(raw, project_root=local_lab3_root)
    if resume_remote_path:
        existing_fingerprint = _existing_run_fingerprint(local_lab3_root, started_day, run_name)
        if existing_fingerprint:
            cfg = replace(cfg, config_fingerprint=existing_fingerprint)
    remote_cfg = replace(cfg, project_root=Path(str(REMOTE_LAB3_ROOT))).with_training_updates(
        data_root=Path(str(REMOTE_DATA_ROOT)),
        artifact_root=Path(str(REMOTE_RUNS_VOLUME_ROOT / "runs")),
        modal_app=APP_NAME,
    )

    payload: dict[str, Any] = {
        "config": remote_cfg.as_json(),
    }
    if resume_remote_path:
        payload["resume_remote_path"] = resume_remote_path
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

    with app.run(detach=detach):
        call = remote_fn.spawn(payload)
        if detach:
            return {
                "run_root": str(local_lab3_root / "runs" / started_day / run_name),
                "teacher_model_version": cfg.teacher_model_version,
                "config_fingerprint": cfg.config_fingerprint,
                "execution": {
                    "backend": "modal",
                    "final_status": "detached",
                    "modal_app_name": APP_NAME,
                    "modal_function_name": FUNCTION_NAME,
                    "call_id": getattr(call, "object_id", ""),
                    "resume_remote_path": resume_remote_path,
                    "data_volume_sync": data_sync,
                },
            }
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
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    }
                )
            except Exception:
                raise

    summary = finalize_modal_result(
        cfg=cfg,
        local_lab3_root=local_lab3_root,
        local_data_root=local_data_root,
        runs_volume_name=runs_volume_name,
        fallback_summary=fallback_summary,
        heartbeats=heartbeats,
        was_cut_off=was_cut_off,
    )
    summary.setdefault("execution", {})["data_volume_sync"] = data_sync
    return summary


def main_cli() -> None:
    import argparse
    import yaml

    p = argparse.ArgumentParser(description="Launch Restormer teacher training on Modal (optional local helper).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--started-day", type=str, default="")
    p.add_argument("--resume-remote-path", type=str, default="")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--detach", action="store_true")
    p.add_argument("--sync-data", action="store_true")
    p.add_argument("--force-data-sync", action="store_true")
    p.add_argument("--gpu", type=str, default=DEFAULT_GPU)
    p.add_argument("--timeout-minutes", type=int, default=DEFAULT_TIMEOUT_MINUTES)
    p.add_argument("--modal-data-volume", type=str, default=DEFAULT_DATA_VOLUME_NAME)
    p.add_argument("--modal-runs-volume", type=str, default=DEFAULT_RUNS_VOLUME_NAME)
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw.setdefault("training", {})["config_path"] = str(cfg_path)

    started_day = args.started_day or time.strftime("%Y-%m-%d", time.gmtime())
    local_lab3 = _LOCAL_LAB3_ROOT
    local_data = (local_lab3 / "Data").resolve()
    out = execute_restormer_teacher_modal(
        raw,
        local_lab3_root=local_lab3,
        local_data_root=local_data,
        run_name=args.run_name,
        started_day=started_day,
        resume_remote_path=args.resume_remote_path,
        smoke_test=args.smoke_test,
        detach=args.detach,
        sync_data=args.sync_data,
        force_data_sync=args.force_data_sync,
        gpu=args.gpu,
        timeout_minutes=args.timeout_minutes,
        data_volume_name=args.modal_data_volume,
        runs_volume_name=args.modal_runs_volume,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main_cli()
