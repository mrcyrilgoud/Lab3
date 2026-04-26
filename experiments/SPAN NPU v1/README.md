# SPAN NPU v1

Self-contained Lab 3 port of the [Swift Parameter-free Attention Network (SPAN, CVPRW 2024)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wan_Swift_Parameter-free_Attention_Network_for_Efficient_Super-Resolution_CVPRW_2024_paper.pdf) adapted to the Lab 3 `256x256x3 -> 256x256x3` restoration contract and the Mobilint MLA-100 operator envelope.

## Scope

This folder contains a single self-contained notebook and this README. All model, training, REP collapse, ONNX export, calibration, operator audit, MXQ handoff, and Modal dispatch code lives inside the notebook cells per the rubric's [Notebook Independence](/Users/mrcyrilgoud/Desktop/repos/Lab3/docs/model_notebook_requirements_rubric.md) rule. There are no companion `.py` modules.

- Notebook: [lab3_span_npu_v1_modal.ipynb](lab3_span_npu_v1_modal.ipynb)
- Rubric: [docs/model_notebook_requirements_rubric.md](/Users/mrcyrilgoud/Desktop/repos/Lab3/docs/model_notebook_requirements_rubric.md)
- Plan: `.cursor/plans/span_npu_investigation_e8337ac2.plan.md`

## Architecture (primary variant)

```text
x [N,3,256,256]
  -> Conv3x3(3->C) + LeakyReLU                                  # stem
  -> [SPAB(C)] x 6                                              # body
        H = RepConv3x3 -> LeakyReLU -> RepConv3x3 -> LeakyReLU -> RepConv3x3
        U = H + input
        V = HardSigmoid(H) - 0.5                                # parameter-free attention
        O = U * V
  -> sum(O_0, O_1, O_5, Conv3x3(O_6))                           # NPU-safe aggregation
  -> Conv3x3(C->3)                                              # tail
  -> delta
y = x + delta
```

At export the `RepConv3x3` collapses to a single `Conv3x3` per RepVGG.

## Config Flags (defaults in bold)

| Flag | Values | Notes |
| --- | --- | --- |
| `channels` | **28**, 48 | C=48 is a post-winner follow-up |
| `num_blocks` | **6** | Paper default |
| `attention` | `faithful`, **`hardsigmoid`**, `hardswish`, `none` | `hardsigmoid` primary |
| `aggregation` | `concat`, **`sum`**, `sequential` | `sum` avoids Concatenate fallback |
| `use_rep` | **true**, false | RepVGG-style training branches |
| `use_teacher` | true, **false** | Toggle Restormer teacher distillation |
| `teacher_weight` | float, default **0.0** | L1 teacher loss weight |

## Parameter Targets

- `C=28`, sum-aggregation, residual tail: approx **143K** params.
- `C=48`, sum: approx **418K**; `C=48`, concat: approx **480K**.

## NPU Operator Audit Summary

- Preferred on MLA-100 in use: `Conv2d`, `LeakyReLU`, `HardSigmoid`, `HardSwish`, `Add` (global residual).
- CPU fallback risk depending on variant: `Sigmoid` (faithful only), `Sub` (attention `-0.5`), `Multiply` (attention gate), `Concatenate` (concat aggregation only).
- REP collapses to a single `Conv3x3` per slot at export; fully NPU-safe at inference.

## Experiment Matrix

| Variant | `attention` | `aggregation` | `use_teacher` | Purpose |
| --- | --- | --- | --- | --- |
| `span_default` | `hardsigmoid` | `sum` | false | Primary: NPU-friendly gate + safe aggregation |
| `span_faithful` | `faithful` | `concat` | false | Paper-exact ceiling + fallback cost instrumentation |
| `span_hardswish` | `hardswish` | `sum` | false | HardSwish self-gate (fused activation + gate) |
| `span_noatt` | `none` | `sum` | false | SPAN-noatt ablation, fully NPU-safe |
| `span_default_teacher` | `hardsigmoid` | `sum` | **true** | Winner + Restormer teacher distillation |
| `span_large` | (winner) | (winner) | (winner) | C=48 rerun of the winner |

Each variant runs for a 2-hour Modal wallclock on `lab3-data` + `lab3-runs`, logs to `runs/<YYYYMMDD>/<run_name>/`, and registers a ledger entry under `runs/autopilot_reports/` with a stable config hash so same-slice comparison to `WideResidualNoBN` is well-defined.

## NPU Latency Workflow

NPU latency measurement requires a Mobilint MLA-100 host and is NOT available on Modal. The notebook ships two supporting cells:

- `measure_npu_latency(mxq_path, variant, candidate_id, ...)` - runs warmup + timed inference against the compiled `.mxq`, writes `runs/<day>/<run>/exports/npu_latency.json`. No-ops gracefully when the `maccel` runtime is not importable or the `.mxq` is missing.
- `aggregate_npu_latency(variants)` - scans `runs/<day>/<run>/exports/npu_latency.json` for each variant and writes `runs/autopilot_reports/span_npu_latency.json`.

Runbook on a Mobilint-equipped host, after the Modal sweep has synced back the exports:

```bash
# One-time per variant (on the NPU host):
python '/Users/mrcyrilgoud/Desktop/repos/Lab3/lab3_step2_onnx_to_mxq.py' \
  --onnx  runs/<day>/<run>/exports/span_npu_v1.onnx \
  --output runs/<day>/<run>/exports/span_npu_v1.mxq \
  --calibration-dir runs/<day>/<run>/exports/calibration

# Then re-run the lab3_span_npu_v1_modal.ipynb notebook on the NPU host with
# LAB3_SPAN_RUN_ROOT=runs/<day>/<run> to have the notebook's NPU latency cell
# produce runs/<day>/<run>/exports/npu_latency.json, followed by the aggregator
# cell which updates runs/autopilot_reports/span_npu_latency.json.
```

## Non-goals

- `downup` internal stride-2 + `PixelShuffle(2)` task adaptation (SPAN SR flavour).
- L1 + L2 fine-tune stage (paper Section 4 stage 2).
- Local training (Modal only, per autopilot rules).
- Committing, pushing, or opening PRs (forbidden by autopilot rules).
