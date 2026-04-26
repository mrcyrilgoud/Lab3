#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    default_cfg = _pkg_root() / "configs" / "restormer_teacher.yaml"
    p = argparse.ArgumentParser(description="Train Restormer-style teacher (same-res 256 restoration).")
    p.add_argument("--config", type=str, default=str(default_cfg), help="Path to restormer_teacher.yaml")
    p.add_argument("--data-root", type=str, default="", help="Override data root (default from YAML / Data)")
    p.add_argument("--run-name", type=str, default="", help="Override run name in YAML")
    p.add_argument("--smoke-test", action="store_true", help="Smoke: smoke profile, 1 epoch, small batch")
    p.add_argument("--resume", type=str, default="", help="Path to latest.pth or best_ema.pth checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pkg = _pkg_root()
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

    from restormer_teacher.config import resolve_teacher_config
    from restormer_teacher.train import run_training

    project_root = _project_root()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tr = raw.setdefault("training", {})
    tr["config_path"] = str(cfg_path)
    if args.data_root:
        tr["data_root"] = args.data_root
    if args.run_name:
        tr["run_name"] = args.run_name
    if not tr.get("started_day"):
        tr["started_day"] = os.environ.get("MODAL_RUN_DAY", "") or time.strftime("%Y-%m-%d", time.gmtime())
    if args.smoke_test:
        tr["smoke_test_override"] = True
    resume = Path(args.resume).resolve() if args.resume else None

    cfg = resolve_teacher_config(raw, project_root=project_root)
    run_training(cfg, resume_path=resume)


if __name__ == "__main__":
    main()
