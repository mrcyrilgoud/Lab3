# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

DATA255 Lab 3: build a from-scratch PyTorch super-resolution model that hits the contract `256x256x3 → 256x256x3`, exports to ONNX, converts to MXQ via the supplied script, and runs on the Mobilint MLA-100 NPU. Grading is mean PSNR (`> 25 dB` threshold) and NPU latency on a hidden test set; if the model cannot run on the NPU, the score is zero.

The canonical standard for *all* Lab 3 work in this repo is [docs/notes/model_notebook_requirements_rubric.md](docs/notes/model_notebook_requirements_rubric.md). Read it before editing notebooks, training code, ONNX export, calibration, or conversion logic. [AGENTS.md](AGENTS.md) restates the same hard rules (note: the absolute paths in AGENTS.md point at `/Users/mrcyrilgoud/...`, but this repo lives at `/Users/cyrilgoud/Desktop/repos/personal/Lab3/` — use repo-relative paths).

## Hard Constraints (do not violate)

- **I/O contract:** input and output must be `[1, 3, 256, 256]` end-to-end. Never change this shape.
- **NPU-first design:** prefer ops the MLA-100 runs natively (`Conv2d`, `DepthwiseConv`, `GroupConv`, `LeakyReLU`/`PReLU`, `HardSigmoid`/`HardSwish`, `ConvTranspose`). Avoid CPU-fallback ops in the forward path (`Add`/`Mul`/`Concat`/`Sigmoid`/`Reshape`/`Transpose`/`Resize`/`MatMul`/`Sub`/`Clip`/...). The strict allow-list used by some audits (e.g. SRNet) is just `Conv`, `ConvTranspose`, `LeakyRelu` — even a single forbidden op is a hard fail there.
- **Calibration data must be derived from the training split**, not validation, not synthetic.
- **Notebook independence:** Lab 3 deliverable notebooks must be runnable standalone — no imports from repo-local helpers for the required paths (model, data, training, export, calibration). The shared code in [src/](src/) exists for the autopilot scaffolding, not for the submission notebooks.
- **Modal-only training under autopilot.** When running under the [lab3-autopilot skill](skills/lab3-autopilot/SKILL.md) or any autonomous experimentation, do not train locally; use the `lab3-data` and `lab3-runs` Modal volumes.
- **No commits/pushes/PRs from autonomous runs.** Manual commits by the user are fine; the autopilot must not commit.

## Submission Artifact Chain

Every model deliverable produces, in order:

1. `best.pth` (trained weights, EMA preferred when present)
2. `best.onnx` (opset 17, `1x3x256x256` IO, `onnx.checker` + ORT parity check)
3. `calibration/` (training-derived LR images, deterministic seed)
4. `xxx.mxq` (via [ONNX-to-MXQ/lab3_step2_onnx_to_mxq.py](ONNX-to-MXQ/lab3_step2_onnx_to_mxq.py))
5. NPU eval via [ONNX-to-MXQ/lab3_step3_mxq_npu_eval.py](ONNX-to-MXQ/lab3_step3_mxq_npu_eval.py) (only runs on a Mobilint-equipped host; not on Modal)

The reference packaged submission is in [experiments/FSRCNNResidual/submission/](experiments/FSRCNNResidual/submission/) (`lab3.ipynb`, `lab3.onnx`, `lab3.mxq`, `lab3_step2.py`, `lab3_step3.py`).

## Repository Layout

- [experiments/](experiments/) — one subdirectory per model family. Each contains a self-contained notebook (and sometimes a frozen `model.py` snapshot, README, and configs). Examples: `FSRCNNResidual/` (current best, ~24.13 dB val PSNR @ 8000 epochs), `SRNet NPU v1/`, `HiNetLite NPU v1` and `v6/`, `SPAN NPU v1/`, `Restormer NPU v1/`, `Teacher-Student Reformer/` (Restormer teacher distillation), `00_baseline/` (canonical wide-residual no-BN).
- [U-Net Experiment 1/](U-Net%20Experiment%201/) — `SingleBridgeUNet` notebook + Modal launcher. Standalone (mirrors the submission notebook style).
- [src/pipelines/lab3_pipeline_lib.py](src/pipelines/lab3_pipeline_lib.py) — shared training/export/calibration library used by the autopilot scaffolding (not by submission notebooks).
- [src/scripts/](src/scripts/) — autopilot orchestration: `autopilot_controller.py`, `validate_canonical_pipeline.py`, `lab3_modal_app.py`, `run_modal_experiment.py`, `notebook_execution.py`, `lab3_agent_loop.py`, `audit_lab3_data_pipeline.py`.
- [src/utils/](src/utils/) — `generate_lab3_notebook.py` (notebook scaffolding), `lab3_step2_onnx_to_mxq.py` (mirror of the submission script).
- [ONNX-to-MXQ/](ONNX-to-MXQ/) — authoritative copies of the ONNX→MXQ conversion script and NPU eval script.
- [docs/notes/](docs/notes/) — rubric, Lab 2 postmortems, current per-model work notes (e.g. [srnet_current_work.md](docs/notes/srnet_current_work.md)).
- [docs/model_plans/](docs/model_plans/) — design docs per model.
- [docs/references/](docs/references/) — `DATA255_12_lab3-1.pdf` (course handout) and the Mobilint operator reference PDF.
- [skills/lab3-autopilot/](skills/lab3-autopilot/) — repo-local SKILL guiding bounded autonomous experimentation.
- `Data/` and `runs/` are gitignored. `Data/` contains `HR_train/HR_train{1..4}`, `LR_train/LR_train{1..4}`, `HR_val`, `LR_val` paired by basename. `runs/<YYYY-MM-DD>/<run_name>/` is where Modal sync-back lands and is the source of truth for trained artifacts.

## Common Commands

Validate the canonical Modal pipeline (smoke test, requires Modal SDK + auth):

```bash
python3 src/scripts/validate_canonical_pipeline.py \
  --notebook experiments/00_baseline/lab3_wide_residual_nobn_modal_app.ipynb
```

Bounded autopilot smoke run (forces a known candidate, records to `runs/autopilot_reports/`):

```bash
python3 src/scripts/autopilot_controller.py \
  --max-runs 1 \
  --force-candidate wide_residual_nobn_v1 \
  --rerun-reason "real Modal smoke test" \
  --train-pairs 8 --val-pairs 4 \
  --num-epochs 1 --warmup-epochs 1 \
  --budget-minutes-per-run 10
```

ONNX → MXQ conversion (run on a Mobilint host, after sync-back from Modal):

```bash
python3 ONNX-to-MXQ/lab3_step2_onnx_to_mxq.py \
  --onnx runs/<day>/<run>/exports/<model>.onnx \
  --output runs/<day>/<run>/exports/<model>.mxq \
  --calibration-dir runs/<day>/<run>/exports/calibration
```

Audit the Data/ pairing layout:

```bash
python3 src/scripts/audit_lab3_data_pipeline.py
```

There is no test suite; the "tests" are: rubric checklist passes, ONNX checker passes, ORT parity `<1e-3`, operator audit clean against the MLA whitelist, and a real bounded Modal run succeeds end-to-end.

## Modal Conventions

- App names follow `lab3-<experiment>-<purpose>` (e.g. `lab3-modal-pipeline`, `lab3-unet-experiment-1`).
- Volumes: `lab3-data` (mounted at `/mnt/lab3-data`, with `Data/` inside) and `lab3-runs` (mounted at `/mnt/lab3-runs`).
- Image Python version is set from the local notebook kernel — keep notebook kernel and Modal `python_version` aligned for notebook-defined Modal functions.
- After a run, sync results back into `runs/<YYYY-MM-DD>/<run_name>/` locally; `latest_status.json`, `summary.json`, `report.json`, `metrics.jsonl`, `run_config.json` are the standard files to read.

## Autopilot Reporting

Autopilot writes three files under [runs/autopilot_reports/](runs/autopilot_reports/) (gitignored):

- `ledger.jsonl` — append-only run history (one JSON per line). Each entry has a stable config hash; duplicate configs are skipped unless `rerun_reason` is set.
- `best_known.json` — current best comparable run.
- `inbox_summary.md` — human-readable handoff snapshot (best PSNR, ΔPSNR, artifact readiness for `.pth`/`.onnx`/calibration/MXQ, next mutation).

Same-slice, same-budget comparisons only — do not promote a run that did not complete or that was scored on a different data slice / epoch budget.

## Current Reference Numbers

Best validation PSNRs to beat (as of [docs/notes/srnet_current_work.md](docs/notes/srnet_current_work.md)):

- FSRCNN 96/40/m8 @ 8000 epochs: **24.1335** (current strongest submission candidate)
- FSRCNN 96/40/m8 @ 6000 epochs: 24.0841
- SRNet recovery best: 24.0179
- HiNetLite NPU v6: 23.9301
- Course handout baseline: 23.1 val / 23.3 hidden

The PSNR bar that unlocks the scoring table is `> 25 dB`; below that the score is capped at 8 regardless of latency. Once over 25 dB, latency becomes the differentiator.
