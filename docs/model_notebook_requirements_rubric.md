# Model Notebook Requirements and Review Rubric

This document is the standard for all Lab 3 model notebooks in this repository. It converts the course handout, the Mobilint MLA-100 operator support notes, the repository data layout, and the Lab 2 postmortems into a single review guide for notebook creation.

## Scope

Use this document for any notebook that does one or more of the following:

- defines a candidate Lab 3 model
- trains or fine-tunes a model
- exports a model to ONNX
- prepares calibration data
- converts ONNX to MXQ
- reports validation or NPU results

This is a practical checklist for building Lab 3 notebooks that are complete and ready for submission.

## Source Basis

- `DATA255_12_lab3-1.pdf`
- `bence.pasuko.com_mla-supported-operations_.pdf`
- `docs/lab2_comprehensive_review.md`
- `docs/lab2_what_worked_and_next_steps.md`
- repository data layout under `Data/`

## Core Course Requirements

These are non-negotiable because they come directly from the course handout.

- Submission is due by `3:00 PM` on `April 29, 2026`.
- Work may be discussed collaboratively, but code and solutions must be written independently.
- The submitted model must be a PyTorch super-resolution model built from scratch.
- Model input format must be `[256, 256, 3]`.
- Model output format must be `[256, 256, 3]`.
- The notebook must produce a trained weight file such as `xxx.pth`.
- The submission must include an `ONNX -> MXQ` conversion script.
- The submission must include an `xxx.mxq` file.
- Calibration data for quantization must come from the training dataset.
- The model must execute on the NPU. If it does not execute on NPU, the score is zero.
- Lab evaluation is based on mean PSNR on the hidden test set, measured on NPU, with latency also affecting score.

## Course Score Target

The lab handout’s third page is labeled `Grading criteria for lab2`, but it appears inside `DATA255_12_lab3-1.pdf`. Treat it as the operative Lab 3 grading table unless the instructor says otherwise.

| Test-set mean PSNR on NPU | Test-set mean latency on NPU | Course score |
|---|---:|---:|
| `> 25 dB` | `< 6 ms` | 18 |
| `> 25 dB` | `< 8 ms` | 17 |
| `> 25 dB` | `< 10 ms` | 16 |
| `> 25 dB` | `< 12 ms` | 15 |
| `> 25 dB` | `< 14 ms` | 14 |
| `> 25 dB` | `< 16 ms` | 13 |
| `> 25 dB` | `< 18 ms` | 12 |
| `> 25 dB` | `< 20 ms` | 11 |
| `> 25 dB` | `> 20 ms` | 10 |
| `<= 25 dB` | `Any` | 8 |

Reference numbers from the handout:

- validation baseline mean PSNR: `23.1 dB`
- hidden test baseline mean PSNR: `23.3 dB`
- practical target: exceed `25 dB`, then reduce latency

## Repository Data Assumptions

Every notebook should explicitly validate the data layout it expects before training begins.

- paired training HR: `Data/HR_train/HR_train1` through `Data/HR_train/HR_train4`
- paired training LR: `Data/LR_train/LR_train1` through `Data/LR_train/LR_train4`
- paired validation HR: `Data/HR_val`
- paired validation LR: `Data/LR_val`
- optional auxiliary data: `Data/course_files_export/`

Minimum notebook behavior:

- verify all required paths exist
- verify LR/HR pairing by basename
- print pair counts for train and validation
- fail fast if no training pairs or validation pairs are found
- keep the notebook self-contained so it can run by itself without importing repo-local helper modules

## Hard Gates

A notebook is not submission-ready if any of the following is missing.

- model definition and training code
- trained weight file such as `.pth`
- ONNX export
- training-derived calibration data
- ONNX-to-MXQ conversion script or exact handoff to it
- final artifact paths for `.pth`, `.onnx`, and `.mxq`
- evidence that the model uses the required `256x256x3` input/output format
- evidence that the model can run through the NPU deployment pipeline

Any notebook that stops before export and packaging should be treated as incomplete.

## Notebook Independence

Every Lab 3 notebook should be runnable on its own.

- include the model, data loading, training, export, and artifact logic inside the notebook itself
- do not depend on repo-local helper modules for required execution paths
- if a companion script is required for submission, the notebook should either generate it directly or contain the exact code needed to recreate it

## Required Notebook Structure

Keep the notebook structure minimal and requirement-driven. These sections are enough.

### 1. Setup

Must include:

- notebook title and model name
- imports and device setup
- key config values
- data path checks

### 2. Data

Must include:

- how LR/HR pairs are loaded
- train and validation counts
- confirmation that the model uses `256x256x3` input/output

### 3. Model and Training

Must include:

- model definition
- brief architecture summary
- training loop
- saved trained weight file

### 4. Validation

Must include:

- validation PSNR or equivalent validation summary
- enough evidence that the model output is better than or meaningfully different from raw LR input

### 5. Export and Submission Artifacts

Must include:

- ONNX export
- ONNX parity or export sanity check
- calibration export from training-derived data
- ONNX-to-MXQ conversion script path or compile cell
- final artifact paths for `.pth`, `.onnx`, and `.mxq`

## NPU Design Rules

The Mobilint support notes make one rule obvious: supported-on-paper is not the same as fast-on-NPU. CPU fallback ops are functionally dangerous because they usually destroy latency.

### Preferred Forward-Path Building Blocks

These are the safest default choices for Lab 3 notebooks.

- `Convolution`
- `DepthwiseConvolution`
- `GroupConvolution`
- `TransposeConvolution` only if truly needed
- `PRelu` or `LeakyReLU` style activations
- `HardSigmoid` or `HardSwish` when an attention gate is unavoidable
- `Pad`
- `Pooling`
- `DepthToSpace` or `Upsampling` only when architecture requires them
- plain residual additions only if they compile cleanly in the actual pipeline

### Avoid In the Main Forward Path

These were listed as `CPU Fallback` in the MLA-100 support sheet and should be treated as latency risks.

- `Adding` / `AddingConstant`
- `Cast`
- `Clip`
- `Concatenate`
- `Div`
- `Flatten`
- `Gather` / `GatherND`
- `Gelu`
- `GroupNormalization`
- `MatMul`
- `Multiply`
- `ReduceMean` and other `Reduce*` ops
- `Relu`
- `Reshape`
- `Resize`
- `Sigmoid`
- `Slice`
- `Softmax`
- `Split`
- `Squeeze`
- `Sub`
- `Tanh`
- `Tile`
- `Transpose`
- `Unsqueeze`
- `Where`

### Do Not Introduce Without Strong Justification

These were documented as unsupported or compiler-breaking in the supplied operator notes.

- `Abs`
- `Ceil`
- `Conv3d`
- `CumSum`
- `Flip`
- `Floor`
- `GRU`, `LSTM`, `RNN`
- `GatherElements`
- `GridSample`
- `LayerNormalization`
- `Less`, `LessOrEqual`
- `Log`, `LogSigmoid`, `Logit`
- element-wise `Maximum` / `Minimum`
- `NonMaxSuppression`
- `NonZero`
- `Or`, `Xor`
- `Reciprocal`
- `RmsNormalization`
- `Sin`
- `SpaceToDepth`
- `Sqrt`
- `Tril`

## Architecture Guidance From Lab 2

The Lab 2 repository history provides practical constraints that should shape every new Lab 3 notebook.

- optimize for deployment simplicity first, then add complexity only with evidence
- assume latency matters as much as accuracy once PSNR clears the threshold
- avoid BatchNorm by default for this workload unless a measured benefit justifies it
- avoid large numbers of normalization, attention, gating, and fallback-prone ops
- prefer residual formulations that learn a correction over the LR input
- verify the model is improving over identity on validation before spending time on export
- treat synthetic data as optional and evidence-driven, not automatically beneficial

Practical default bias for candidate architectures:

- fully convolutional
- static spatial resolution
- small operator vocabulary
- no dynamic control flow
- no data-dependent output shapes
- minimal post-processing outside the network

## Simple Rubric

Score notebooks out of `100` using the course-facing essentials.

| Category | Points | What earns full credit |
|---|---:|---|
| Required format and data usage | 20 | Notebook clearly uses the required paired data and confirms `256x256x3` input/output |
| Model and training | 20 | Model architecture and training code are present and produce a trained weight file |
| Validation evidence | 15 | Validation PSNR or equivalent result summary is reported clearly |
| ONNX export | 15 | Model exports to ONNX and includes an export sanity check or parity check |
| Calibration and MXQ path | 15 | Calibration data comes from training-derived data and the ONNX-to-MXQ step is clearly defined |
| Final submission artifacts | 15 | Final paths for `.pth`, `.onnx`, and `.mxq` are provided clearly |

## Automatic Fail Conditions

If any of these are true, the notebook is incomplete.

- model input or output shape is not `256x256x3`
- notebook cannot locate the required paired data
- model definition or training code is missing
- trained weight file is missing
- ONNX export is missing
- calibration data does not come from training-derived data
- ONNX-to-MXQ conversion step is missing
- final `.mxq` artifact or exact handoff path is missing

## Suggested Artifact Layout

Use stable names so artifacts are easy to compare across notebooks.

- notebook: `lab3_<model_id>_<purpose>.ipynb`
- run name: timestamped, for example `20260411_153000`
- checkpoints: `runs/<run_name>/checkpoints/`
- exports: `runs/<run_name>/exports/`
- calibration: `runs/<run_name>/exports/calibration/`
- summary file: `runs/<run_name>/latest_status.json` or equivalent

## Final Checklist

Before submission, confirm all of the following.

- training and validation both ran end to end
- best validation PSNR is reported
- improvement over LR baseline is reported
- operator audit is printed
- ONNX export succeeded
- ONNX parity check passed
- calibration assets exist
- MXQ compile handoff is explicit
- all final artifact paths are printed in one place

## Copyable Notebook Skeleton

Use this as the minimum outline for future model notebooks.

```markdown
# Lab 3 - <model_id>

- Author: <name>
- Date: <YYYY-MM-DD>
- Goal: improve PSNR | reduce latency | both
- Expected input/output: 256x256x3

## 1. Setup
- imports
- device resolution
- key config
- data path checks

## 2. Data
- data-root validation
- train/val pair counts
- input/output size confirmation

## 3. Model and Training
- architecture definition
- loss
- optimizer
- training loop
- checkpointing

## 4. Validation
- best val PSNR
- brief validation summary

## 5. Export and Submission Artifacts
- ONNX export
- ONNX parity check
- training-derived calibration export
- conversion script path
- expected input/output artifact paths
- best .pth
- .onnx
- .mxq
```

## References

- [DATA255_12_lab3-1.pdf](/Users/mrcyrilgoud/Desktop/repos/Lab3/DATA255_12_lab3-1.pdf)
- [bence.pasuko.com_mla-supported-operations_.pdf](/Users/mrcyrilgoud/Desktop/repos/Lab3/bence.pasuko.com_mla-supported-operations_.pdf)
- [lab2_comprehensive_review.md](/Users/mrcyrilgoud/Desktop/repos/Lab3/docs/lab2_comprehensive_review.md)
- [lab2_what_worked_and_next_steps.md](/Users/mrcyrilgoud/Desktop/repos/Lab3/docs/lab2_what_worked_and_next_steps.md)
