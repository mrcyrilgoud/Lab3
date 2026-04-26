from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


PKG_ROOT = Path(__file__).resolve().parents[1]
LAB3_ROOT = PKG_ROOT.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from restormer_teacher.checkpointing import is_legacy_teacher_checkpoint, save_checkpoint
from restormer_teacher.config import TEACHER_MODEL_VERSION, resolve_teacher_config
from restormer_teacher.logging_utils import load_jsonl
from restormer_teacher.model import BiasFreeLayerNorm, WithBiasLayerNorm
from restormer_teacher.run_state import reconcile_run_state
from tools.restormer_teacher_modal_app import finalize_modal_result


def _raw_cfg() -> dict:
    return {
        "active_profile": "large",
        "profiles": {
            "smoke": {
                "dim": 32,
                "num_blocks": [2, 2, 4, 2],
                "num_refinement_blocks": 2,
                "heads": [1, 2, 4, 8],
                "ffn_expansion_factor": 2.66,
            },
            "large": {
                "dim": 48,
                "num_blocks": [4, 6, 6, 8],
                "num_refinement_blocks": 4,
                "heads": [1, 2, 4, 8],
                "ffn_expansion_factor": 2.66,
            },
        },
        "training": {
            "data_root": "Data",
            "artifact_root": "runs",
            "run_name": "teacher_test",
            "started_day": "2026-04-15",
            "epochs": 3,
            "batch_size": "auto",
            "lr": 2.0e-4,
            "min_lr": 1.0e-6,
            "weight_decay": 1.0e-4,
            "warmup_epochs": 3,
            "patch_size": 192,
            "ema_decay": 0.999,
            "grad_clip": 1.0,
            "seed": 42,
            "train_num_workers": 0,
            "val_num_workers": 0,
            "log_step_interval": 0,
            "config_path": str(PKG_ROOT / "configs" / "restormer_teacher.yaml"),
        },
    }


def _epoch_record(epoch: int) -> dict:
    return {"kind": "epoch", "epoch": epoch, "train_loss": 1.0 - epoch * 0.1}


class TestLayerNorms(unittest.TestCase):
    def test_biasfree_layernorm_matches_variance_only_scaling(self) -> None:
        x = torch.tensor([[[[1.0]], [[3.0]]]])
        norm = BiasFreeLayerNorm(2)
        with torch.no_grad():
            norm.weight.fill_(1.0)
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        expected = x / torch.sqrt(var + norm.eps)
        out = norm(x)
        self.assertTrue(torch.allclose(out, expected))

    def test_withbias_layernorm_still_centers_and_biases(self) -> None:
        x = torch.tensor([[[[1.0]], [[3.0]]]])
        norm = WithBiasLayerNorm(2)
        with torch.no_grad():
            norm.weight.fill_(1.0)
            norm.bias.fill_(0.5)
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        expected = (x - mean) / torch.sqrt(var + norm.eps) + 0.5
        out = norm(x)
        self.assertTrue(torch.allclose(out, expected))


class TestRunState(unittest.TestCase):
    def test_reconcile_fresh_run_accepts_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            state = reconcile_run_state(
                run_root=run_root,
                history_meta={"run_id": "fresh", "teacher_model_version": TEACHER_MODEL_VERSION, "config_fingerprint": "abc"},
                resume_epoch=None,
                allow_legacy_checkpoint=False,
            )
            self.assertIsNone(state.history)
            self.assertEqual(state.metrics_records, [])
            self.assertIsNone(state.latest_status)

    def test_reconcile_resume_truncates_future_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            history = {
                "meta": {
                    "run_id": "teacher_test",
                    "teacher_model_version": TEACHER_MODEL_VERSION,
                    "config_fingerprint": "abc",
                },
                "epochs": [_epoch_record(0), _epoch_record(1), _epoch_record(2), _epoch_record(3)],
            }
            (run_root / "history.json").write_text(json.dumps(history), encoding="utf-8")
            metrics_rows = [
                _epoch_record(0),
                {"kind": "train_step", "epoch": 1, "global_step": 7},
                _epoch_record(1),
                _epoch_record(2),
                _epoch_record(3),
            ]
            (run_root / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in metrics_rows),
                encoding="utf-8",
            )
            (run_root / "latest_status.json").write_text(json.dumps({"epoch": 3}), encoding="utf-8")

            reconcile_run_state(
                run_root=run_root,
                history_meta=history["meta"],
                resume_epoch=1,
                allow_legacy_checkpoint=False,
            )
            trimmed_history = json.loads((run_root / "history.json").read_text(encoding="utf-8"))
            self.assertEqual([row["epoch"] for row in trimmed_history["epochs"]], [0, 1])
            trimmed_metrics = load_jsonl(run_root / "metrics.jsonl")
            self.assertEqual([row["epoch"] for row in trimmed_metrics if row.get("kind") == "epoch"], [0, 1])
            self.assertFalse((run_root / "latest_status.json").exists())

    def test_reconcile_detects_history_metric_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            history = {
                "meta": {
                    "run_id": "teacher_test",
                    "teacher_model_version": TEACHER_MODEL_VERSION,
                    "config_fingerprint": "abc",
                },
                "epochs": [_epoch_record(0), _epoch_record(1)],
            }
            (run_root / "history.json").write_text(json.dumps(history), encoding="utf-8")
            (run_root / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in [_epoch_record(0), _epoch_record(2)]),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                reconcile_run_state(
                    run_root=run_root,
                    history_meta=history["meta"],
                    resume_epoch=1,
                    allow_legacy_checkpoint=False,
                )


class TestCheckpointMetadata(unittest.TestCase):
    def test_new_checkpoints_include_teacher_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = torch.nn.Conv2d(3, 3, kernel_size=1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            path = Path(tmpdir) / "latest.pth"
            metadata = {
                "teacher_model_version": TEACHER_MODEL_VERSION,
                "active_profile": "smoke",
                "architecture": {"dim": 32},
                "config_path": "cfg.yaml",
                "config_fingerprint": "abc123",
                "run_name": "teacher_test",
                "started_day": "2026-04-15",
            }
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler_state=None,
                epoch=0,
                global_step=1,
                best_val_psnr_ema=1.0,
                ema_state={"decay": 0.9, "shadow": {}},
                teacher_metadata=metadata,
                extra={},
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["teacher"]["teacher_model_version"], TEACHER_MODEL_VERSION)
            self.assertFalse(is_legacy_teacher_checkpoint(ckpt))
            self.assertEqual(ckpt["teacher"]["config_fingerprint"], "abc123")

    def test_unversioned_checkpoint_is_legacy(self) -> None:
        ckpt = {"model": {}, "optimizer": {}, "ema": {}}
        self.assertTrue(is_legacy_teacher_checkpoint(ckpt))


class TestModalFinalization(unittest.TestCase):
    def test_timeout_without_remote_run_returns_structured_timeout(self) -> None:
        cfg = resolve_teacher_config(_raw_cfg(), project_root=LAB3_ROOT)
        with mock.patch("tools.restormer_teacher_modal_app.remote_run_exists", return_value=False):
            summary = finalize_modal_result(
                cfg=cfg,
                local_lab3_root=LAB3_ROOT,
                local_data_root=LAB3_ROOT / "Data",
                runs_volume_name="lab3-runs",
                fallback_summary=None,
                heartbeats=[{"status": "still_running"}],
                was_cut_off=True,
            )
        self.assertEqual(summary["execution"]["final_status"], "cut_off")
        self.assertFalse(summary["artifact_readiness"]["history_ready"])

    def test_partial_remote_run_syncs_when_present(self) -> None:
        cfg = resolve_teacher_config(_raw_cfg(), project_root=LAB3_ROOT)
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            (run_root / "latest_status.json").write_text(json.dumps({"epoch": 0}), encoding="utf-8")
            (run_root / "run_config.json").write_text(json.dumps({"teacher_model_version": TEACHER_MODEL_VERSION}), encoding="utf-8")
            (run_root / "history.json").write_text(
                json.dumps({"meta": {"teacher_model_version": TEACHER_MODEL_VERSION}, "epochs": [_epoch_record(0)]}),
                encoding="utf-8",
            )
            with (
                mock.patch("tools.restormer_teacher_modal_app.remote_run_exists", return_value=True),
                mock.patch("tools.restormer_teacher_modal_app.sync_run_from_volume", return_value=run_root),
                mock.patch("tools.restormer_teacher_modal_app.normalize_synced_teacher_run", return_value=None),
            ):
                summary = finalize_modal_result(
                    cfg=cfg,
                    local_lab3_root=LAB3_ROOT,
                    local_data_root=LAB3_ROOT / "Data",
                    runs_volume_name="lab3-runs",
                    fallback_summary={"remote": "summary"},
                    heartbeats=[],
                    was_cut_off=True,
                )
        self.assertEqual(summary["execution"]["final_status"], "cut_off")
        self.assertEqual(summary["local_run_root"], str(run_root))
        self.assertTrue(summary["artifact_readiness"]["history_ready"])


if __name__ == "__main__":
    unittest.main()
