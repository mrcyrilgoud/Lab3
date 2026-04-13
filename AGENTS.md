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

## Expected Behavior

When working on Lab 3 tasks:

- read the rubric first before making changes
- follow the required notebook structure and hard gates in the rubric
- keep notebooks aligned with the Lab 3 input/output requirement of `256x256x3`
- keep calibration data derived from training data
- ensure deliverables stay aligned with the `.pth`, `.onnx`, conversion-script, and `.mxq` submission chain
- prefer notebook designs and model choices that are compatible with the NPU deployment path described in the rubric

## Priority

If a Lab 3 request conflicts with older repo habits or earlier notebook patterns, prefer the rubric unless the user explicitly asks otherwise.
