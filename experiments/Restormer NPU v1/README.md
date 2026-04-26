# Restormer NPU v1

This folder is a frozen v1 snapshot of the baseline Restormer audit/export path for Lab 3.

- It is a baseline Restormer snapshot, not a new architecture family.
- It is an audit/export package for ONNX, operator-risk review, calibration export, and MXQ handoff.
- The canonical teacher training path remains under [`Teacher-Student Reformer/`](/Users/mrcyrilgoud/Desktop/repos/Lab3/Teacher-Student%20Reformer).

## Purpose

Use this folder when you want a self-contained notebook/runtime that:

- does not import from `Teacher-Student Reformer/` at notebook runtime
- preserves the required `256x256x3` input/output contract
- exports ONNX for the current baseline teacher architecture
- writes training-derived calibration inputs
- records a dry-run MXQ handoff payload
- documents the operator-risk picture for NPU deployment

## Scope

This folder is intentionally narrow:

- `model.py` is the frozen baseline model snapshot for v1.
- `tools/audit_support.py` provides the minimum local support needed for config loading, data validation, ONNX export, parity checks, calibration export, and MXQ handoff.
- `notebooks/lab3_restormer_npu_v1_modal.ipynb` is the main audit notebook.

## Conventions

- Artifact root: `runs/<started_day>/<run_name>/exports/restormer_npu_v1/`
- ONNX artifact: `restormer_npu_v1.onnx`
- MXQ target artifact: `restormer_npu_v1.mxq`
- Modal naming stays aligned with the repo defaults: `lab3-data` and `lab3-runs`

## Non-goals

- This is not the primary Lab 3 submission notebook.
- This does not replace the canonical teacher package.
- This does not implement architecture sweeps such as InstanceNorm, CELU, unit temperature, or conv-mixer variants.
