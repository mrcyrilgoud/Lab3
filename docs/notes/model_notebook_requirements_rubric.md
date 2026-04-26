# Model Notebook Requirements and Review Rubric

Standard for Lab 3 model notebooks here: course handout, Mobilint MLA-100 operator notes, `Data/` layout, and Lab 2 postmortems, in one checklist.

## Scope

Use this for notebooks that define, train, export (ONNX), calibrate, convert (ONNX→MXQ), or report validation/NPU results for a Lab 3 model.

## Source Basis

`DATA255_12_lab3-1.pdf`, `bence.pasuko.com_mla-supported-operations_.pdf`, `docs/lab2_comprehensive_review.md`, `docs/lab2_what_worked_and_next_steps.md`, and `Data/`.

## Core Course Requirements

Non-negotiable (course handout).

- Due `3:00 PM` `April 29, 2026`. Collaboration on ideas is allowed; code and solutions must be independent.
- PyTorch super-resolution model built from scratch; input and output `[256, 256, 3]`.
- Notebook produces trained weights (e.g. `xxx.pth`); submission includes ONNX→MXQ conversion script and `xxx.mxq`.
- Calibration data for quantization must come from the training dataset.
- Model must run on NPU—otherwise course score is zero. Grading uses mean PSNR on the hidden test set (NPU) and NPU latency.

## Course Score Target

Operative Lab 3 grading table (see course PDF; label may read “lab2”).


| Test-set mean PSNR on NPU | Test-set mean latency on NPU | Course score |
| ------------------------- | ---------------------------- | ------------ |
| `> 25 dB`                 | `< 6 ms`                     | 18           |
| `> 25 dB`                 | `< 8 ms`                     | 17           |
| `> 25 dB`                 | `< 10 ms`                    | 16           |
| `> 25 dB`                 | `< 12 ms`                    | 15           |
| `> 25 dB`                 | `< 14 ms`                    | 14           |
| `> 25 dB`                 | `< 16 ms`                    | 13           |
| `> 25 dB`                 | `< 18 ms`                    | 12           |
| `> 25 dB`                 | `< 20 ms`                    | 11           |
| `> 25 dB`                 | `> 20 ms`                    | 10           |
| `<= 25 dB`                | `Any`                        | 8            |


Handout baselines: validation `23.1 dB`, hidden test `23.3 dB`. Practical aim: exceed `25 dB` PSNR, then cut latency.

## Repository Data Assumptions

Validate expected layout before training.

- Training HR: `Data/HR_train/HR_train1`–`HR_train4`; training LR: `Data/LR_train/LR_train1`–`LR_train4`.
- Validation HR: `Data/HR_val`; validation LR: `Data/LR_val`.
- Optional: `Data/course_files_export/`.

**Minimum behavior:** paths exist; LR/HR paired by basename; print train/val pair counts; fail fast if counts are zero. Self-contained execution (no repo-local imports for required paths)—see **Notebook Independence**.

## Submission gates (automatic fail)

Incomplete if any of the following:

- Missing or unlocatable required paired data.
- Model definition or training code missing; no trained `.pth`.
- I/O not `256x256x3` or evidence of correct shapes missing.
- No ONNX export; calibration not from training-derived data.
- No ONNX-to-MXQ step and clear handoff; missing `.mxq` or unclear paths for `.pth`, `.onnx`, `.mxq`.
- No evidence the model can run through the NPU deployment pipeline.
- Stops before export and packaging.

## Notebook Independence

Runnable standalone: model, data loading, training, export, and artifact logic live in the notebook; no repo-local helper imports on required paths. If a submission script is required, the notebook generates it or contains the exact code to recreate it.

## Required Notebook Structure

Minimal, requirement-driven sections:


| Section                      | Must include                                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **1. Setup**                 | Title/model name, imports and device, key config, data path checks                                                             |
| **2. Data**                  | How LR/HR pairs load, train/val counts, `256x256x3` I/O confirmation                                                           |
| **3. Model and Training**    | Model definition, short architecture summary, training loop, saved `.pth`                                                      |
| **4. Validation**            | Validation PSNR (or equivalent); evidence output beats (or clearly differs from) raw LR                                        |
| **5. Export and Submission** | ONNX export, parity/sanity check, training-derived calibration, ONNX→MXQ script or cell, final `.pth` / `.onnx` / `.mxq` paths |


## Code Readability

Skimming should reveal what is trained. Short or long notebooks:

- **Architecture is obvious.** Model class and major blocks (stem, residuals, upsample) are easy to find; add a brief shape/forward summary or stage comments—do not scatter the graph across cells.
- **No dead code.** Drop commented experiments, unused helpers, duplicates, and dead cells; keep one authoritative narrative (or a tiny external note).
- **Model first; rest lean.** Data, logging, plots, export stay minimal—reproducible, not a framework.
- **Intent-clear names** for modules, tensors, losses, checkpoints; define non-obvious abbreviations once.
- **Order:** setup → data → model → train → validate → export; group helpers; trim huge or irrelevant cell outputs.

## NPU Design Rules

Supported-on-paper ≠ fast on NPU; ops that fall back to CPU usually wreck latency.

### Preferred forward-path building blocks

Default-safe choices for Lab 3.

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

### Avoid in the main forward path

Listed as `CPU Fallback` on MLA-100—treat as latency risks.

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
- `ReduceMean` and other `Reduce`* ops
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

### Do not introduce without strong justification

Unsupported or compiler-breaking in the supplied operator notes.

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

## Architecture guidance (Lab 2)

- Favor deployment simplicity; add complexity only with evidence. After PSNR clears the bar, latency matters as much as accuracy.
- Skip BatchNorm unless you measure a clear benefit; avoid piles of norm, attention, gating, and fallback-prone ops.
- Prefer residuals that learn a correction over LR; confirm improvement over identity on val before heavy export work.
- Synthetic data: optional and evidence-driven only.

**Candidate shape:** fully convolutional, fixed resolution, small operator set, no dynamic control flow or data-dependent output shapes, minimal extra postprocessing.

## Simple Rubric

Score out of `100` (course-facing essentials).


| Category                       | Points | What earns full credit                                                                        |
| ------------------------------ | ------ | --------------------------------------------------------------------------------------------- |
| Required format and data usage | 20     | Notebook clearly uses the required paired data and confirms `256x256x3` input/output          |
| Model and training             | 20     | Model architecture and training code are present and produce a trained weight file            |
| Validation evidence            | 15     | Validation PSNR or equivalent result summary is reported clearly                              |
| ONNX export                    | 15     | Model exports to ONNX and includes an export sanity check or parity check                     |
| Calibration and MXQ path       | 15     | Calibration data comes from training-derived data and the ONNX-to-MXQ step is clearly defined |
| Final submission artifacts     | 15     | Final paths for `.pth`, `.onnx`, and `.mxq` are provided clearly                              |


## Suggested Artifact Layout

Use stable names so artifacts are easy to compare across notebooks.

- notebook: `lab3_<model_id>_<purpose>.ipynb`
- run name: timestamped, for example `20260411_153000`
- checkpoints: `runs/<run_name>/checkpoints/`
- exports: `runs/<run_name>/exports/`
- calibration: `runs/<run_name>/exports/calibration/`
- summary file: `runs/<run_name>/latest_status.json` or equivalent

## Final Checklist

End-to-end train and val; report best val PSNR and gain over LR baseline; operator audit printed; ONNX export and parity OK; calibration present; MXQ handoff explicit; all artifact paths in one place.

## Copyable Notebook Skeleton

Minimum outline (aligns with **Required Notebook Structure**).

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

