# AGENTS.md

## Scope

These instructions apply to the entire repository.

## Lab 3 Rule

For any Lab 3 work, use [docs/model_notebook_requirements_rubric.md](/Users/mrcyrilgoud/Desktop/repos/Lab3/docs/model_notebook_requirements_rubric.md) as the default standard.

This includes:

- model notebook creation
- notebook edits or reviews
- training code for Lab 3 models
- ONNX export work
- calibration-data preparation
- ONNX-to-MXQ conversion work
- final submission artifact packaging
- autonomous experimentation scaffolding
- autonomous experiment reviews or reruns

## Expected Behavior

When working on Lab 3 tasks:

- read the rubric first before making changes
- follow the required notebook structure and hard gates in the rubric
- keep notebooks aligned with the Lab 3 input/output requirement of `256x256x3`
- keep calibration data derived from training data
- ensure deliverables stay aligned with the `.pth`, `.onnx`, conversion-script, and `.mxq` submission chain
- prefer notebook designs and model choices that are compatible with the NPU deployment path described in the rubric

## Lab 3 Autopilot Rules

For any autonomous experimentation or Codex automation work in this repo:

- read the rubric before making experiment decisions
- treat [lab3_wide_residual_nobn_modal_app.ipynb](/Users/mrcyrilgoud/Desktop/repos/Lab3/lab3_wide_residual_nobn_modal_app.ipynb) as the canonical Modal pipeline entrypoint
- validate the canonical pipeline before running autonomous experiments
- run all training, validation, export, and calibration work on Modal only
- do not run local training
- preserve the required `256x256x3` input/output contract
- keep calibration derived from training data
- use the `lab3-data` Modal volume for dataset access
- sync day-partitioned Modal run outputs back into this repo
- write autopilot reports under [runs/autopilot_reports](/Users/mrcyrilgoud/Desktop/repos/Lab3/runs/autopilot_reports)
- do not rerun identical configs unless a rerun reason is explicitly recorded
- prefer same-slice and same-budget comparisons for ranking decisions
- do not commit, push, or open pull requests as part of autonomous experimentation

## Priority

If a Lab 3 request conflicts with older repo habits or earlier notebook patterns, prefer the rubric unless the user explicitly asks otherwise.
