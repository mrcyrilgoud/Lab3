---
name: lab3-autopilot
description: Modal-only Lab 3 autonomous experimentation policy and reporting scaffold for this repository.
---

# Lab 3 Autopilot

## Primary Goal

Run bounded, rubric-aligned Lab 3 experiments that:

- validate the canonical Modal pipeline first
- explore NPU-safe candidate mutations without breaking the `256x256x3` contract
- avoid duplicate configurations unless a rerun reason is recorded
- keep local reporting current for future Codex automation handoff

## Canonical Entrypoint

- Canonical notebook: [lab3_wide_residual_nobn_modal_app.ipynb](/Users/mrcyrilgoud/Desktop/repos/Lab3/lab3_wide_residual_nobn_modal_app.ipynb)
- Rubric to read first: [docs/model_notebook_requirements_rubric.md](/Users/mrcyrilgoud/Desktop/repos/Lab3/docs/model_notebook_requirements_rubric.md)

## Hard Constraints

- Validate the canonical notebook before autonomous experimentation.
- All training, validation, export, and calibration work must run on Modal only.
- Local training is not allowed.
- Preserve the `256x256x3` input/output contract.
- Calibration must come from training data.
- Dataset access should come from the `lab3-data` Modal volume.
- Day-partitioned Modal outputs must be synced back into the repo.
- Do not commit, push, or open a PR.
- Use explicit TODOs for any unwired logic.

## Experiment Policy

- Start from the canonical no-BN residual family unless an investigation reason is recorded.
- Prefer same-resolution, convolution-first, NPU-safe candidates.
- Keep the starter search space simple and readable.
- Treat benchmark runs as the stable comparable baseline.
- Treat exploration runs as adjacent mutations.
- Treat investigation runs as targeted probes for unresolved questions.

## Candidate Selection Rules

- Prefer the approved no-BN residual baseline for benchmarking.
- Prefer nearby changes first: slightly deeper or slightly wider variants in the same family.
- Prefer mixed-kernel residual variants only when they stay inside the same static-resolution, safe-op envelope.
- Use investigation runs for larger but still NPU-safe probes only after the benchmark path is validated.

## Duplicate Avoidance Rules

- Compute a stable hash from the experiment-defining configuration.
- Skip exact duplicate configs by default.
- Allow a duplicate only when a non-empty rerun reason is explicitly recorded in the ledger entry.

## Comparison Rules

- Prefer same-slice and same-budget comparisons for ranking.
- Do not treat non-comparable runs as automatic promotions.
- Do not treat failed or metricless runs as benchmark wins.

## Promotion Gates

- Benchmark wins require real validation metrics.
- ONNX sanity must be recorded before promotion.
- Artifact readiness must cover `.pth`, `.onnx`, calibration export, and MXQ handoff.
- Placeholder or failed launches must remain non-promoted.
- Use the synced run summary as the source of truth for validation metrics, ONNX sanity, calibration export, and MXQ handoff status.

## Time Budget Policy

- Total wall-clock budget: 4 hours
- Progress review cadence: every 30 minutes
- Stop the active experiment at 4 hours total wall-clock time

## Required Outputs

Write or maintain these files under [runs/autopilot_reports](/Users/mrcyrilgoud/Desktop/repos/Lab3/runs/autopilot_reports):

- `ledger.jsonl`
- `best_known.json`
- `inbox_summary.md`

## Required Inbox Summary Fields

Always include:

- current best comparable validation PSNR
- current best comparable delta PSNR
- whether the best run completed or was cut off
- whether the latest run completed or was cut off
- artifact readiness for `.pth`, `.onnx`, calibration, and MXQ handoff
- most important investigation finding
- next recommended mutation or architecture family
- whether the 4-hour budget was fully used
- whether a longer follow-up run is justified

## Testing

Use the scaffold to prove the automation can make real bounded progress on Modal.

- Run `scripts/validate_canonical_pipeline.py` against the canonical notebook.
- Run `scripts/autopilot_controller.py` with a bounded profile or force a known candidate with a rerun reason for a smoke check.
- Confirm that the controller updates `ledger.jsonl`, `best_known.json`, and `inbox_summary.md`.
- Confirm that the controller either skips duplicate configs or records a new real Modal run entry.
- Confirm that the resulting ledger entry includes real validation metrics when the run completes.
- Confirm that ONNX sanity, calibration export, and MXQ handoff fields are populated from the synced summary.
