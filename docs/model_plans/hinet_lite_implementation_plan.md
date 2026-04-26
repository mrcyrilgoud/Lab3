# HiNet-lite Implementation Plan

## Summary

This document specifies a decision-complete implementation plan for a **HiNet-lite** model for Lab 3. The target is same-resolution image restoration with fixed `256x256x3 -> 256x256x3` I/O, using a **single-stage HiNet-inspired architecture** that preserves the useful restoration bias of HIN blocks while staying disciplined about NPU operator risk and repo integration.

This is **not** a plan for full HiNet. The full paper architecture is intentionally reduced for this repo because the primary Lab 3 constraint is not just validation PSNR; it is also successful export and credible NPU execution under the local Mobilint MLA operator envelope.

## Scope and Rationale

HiNet-lite deliberately removes the highest-risk parts of full HiNet:

- No second U-Net stage
- No supervised attention module
- No cross-stage feature fusion
- No concat-heavy decoder fusion unless export evidence later proves it safe

These reductions are made to cut graph complexity, memory pressure, and fallback-prone operator usage. The goal is to keep the strongest HiNet-inspired idea for this dataset, **half instance normalization in early restoration blocks**, without inheriting the full latency and export burden of the original architecture.

## Architecture Specification

### Model Identity

- Config type: `HiNetLiteConfig`
- Top-level model class: `HiNetLiteSR`
- Reusable modules:
  - `HINResidualBlock`
  - `ResidualBlock`
  - `DownsampleBlock`
  - `UpsampleBlock`
- Candidate id: `hinet_lite_npu_v1`
- Default experiment folder: `experiments/HiNetLite NPU v1/`
- Primary notebook name: `lab3_hinet_lite_npu_v1_modal.ipynb`

### Fixed Contract

- Input tensor: `N x 3 x 256 x 256`
- Output tensor: `N x 3 x 256 x 256`
- Output behavior: predict a residual correction `delta`, then return `y = x + delta`

### Default Configuration

```python
HiNetLiteConfig(
    channels=32,
    encoder_blocks=(2, 2),
    bottleneck_blocks=4,
    decoder_blocks=(2, 2),
    activation="leaky_relu",
    upsample_mode="transpose_conv",
    global_residual=True,
    train_patch_size=224,
    eval_size=256,
)
```

### Top-Level Graph

The model is a single residual U-Net with HIN blocks only in the encoder.

```text
x [N,3,256,256]
  -> stem: Conv3x3(3 -> C) + LeakyReLU
  -> enc1: 2 x HINResidualBlock(C)                      # 256x256
  -> down1: Conv3x3 stride 2 (C -> 2C)                  # 128x128
  -> enc2: 2 x HINResidualBlock(2C)                     # 128x128
  -> down2: Conv3x3 stride 2 (2C -> 4C)                 # 64x64
  -> bottleneck: 4 x ResidualBlock(4C)                  # 64x64
  -> up1: TransposeConv2d (4C -> 2C, stride 2)          # 128x128
  -> fuse1: add skip from enc2 output                   # add-based, channel matched
  -> dec1: 2 x ResidualBlock(2C)                        # 128x128
  -> up2: TransposeConv2d (2C -> C, stride 2)           # 256x256
  -> fuse2: add skip from enc1 output                   # add-based, channel matched
  -> dec2: 2 x ResidualBlock(C)                         # 256x256
  -> tail: Conv3x3(C -> 3) = delta
y = x + delta
```

### Module Definitions

#### `HINResidualBlock`

This block is used only in the encoder path.

Exact definition:

1. `Conv3x3(C -> C)`
2. Split channels into two equal halves
3. Apply `InstanceNorm2d(affine=True)` to the **first half only**
4. Concatenate the normalized half and untouched half
5. `LeakyReLU`
6. `Conv3x3(C -> C)`
7. `LeakyReLU`
8. Residual add with input

Reference shape flow:

```text
input [N,C,H,W]
  -> Conv3x3(C -> C)
  -> split to [N,C/2,H,W] + [N,C/2,H,W]
  -> InstanceNorm on first half only
  -> concat back to [N,C,H,W]
  -> LeakyReLU
  -> Conv3x3(C -> C)
  -> LeakyReLU
  -> add input
output [N,C,H,W]
```

Implementation requirements:

- `channels` must be even; the block should raise if `C % 2 != 0`
- Use one `InstanceNorm2d(C // 2, affine=True, track_running_stats=False)` instance per block
- Do not normalize both halves
- Do not introduce squeeze/excitation, attention gates, or extra mixing branches in this block

#### `ResidualBlock`

Used in the bottleneck and decoder only.

Exact definition:

1. `Conv3x3(C -> C)`
2. `LeakyReLU`
3. `Conv3x3(C -> C)`
4. `LeakyReLU`
5. Residual add with input

Constraints:

- No normalization inside decoder or bottleneck residual blocks for v1
- No channel attention, depthwise branching, gating, or MLP-style subpaths

#### `DownsampleBlock`

Exact definition:

- `Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)`
- `LeakyReLU`

Required stage mapping:

- `down1`: `C -> 2C`
- `down2`: `2C -> 4C`

#### `UpsampleBlock`

Default definition for v1:

- `TransposeConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)`
- `LeakyReLU`

Default choice:

- Use `TransposeConv2d` for v1 because local MLA notes list `TransposeConvolution` as supported
- Do not use resize-based upsampling by default
- `Upsample + Conv` is a future ablation only, not the primary implementation path

### Skip Fusion Rule

Skip fusion is **add-based only** in v1.

- `fuse1`: add decoder feature after `up1` to encoder level 2 output
- `fuse2`: add decoder feature after `up2` to encoder level 1 output

Requirements:

- Decoder and encoder tensors must already have matching channels before addition
- Do not concatenate skip tensors in the forward path for v1
- If a future experiment changes this, it must be justified by export evidence and operator audit results

## Repo Integration Requirements

### Data and Pairing Contract

All implementation work must follow the existing canonical pairing and audit contract in:

- [`docs/notes/hr_lr_pairing_pattern.md`](../notes/hr_lr_pairing_pattern.md)
- [`docs/notes/hr_lr_pipeline_audit_findings.md`](../notes/hr_lr_pipeline_audit_findings.md)

That contract is not optional. Specifically:

- Use canonical `(lr_path, hr_path, name)` tuples
- Run pairing audit before loader creation
- Keep LR and HR transforms synchronized
- Use epoch-aware augmentation seeding for train data
- Preserve the existing full-run count expectations for train and val

### Notebook and Experiment Placement

The implementation should create a self-contained notebook under:

- `experiments/HiNetLite NPU v1/lab3_hinet_lite_npu_v1_modal.ipynb`

The notebook should follow the repo’s existing Lab 3 structure:

- setup
- data checks and pairing audit
- model definition
- training loop
- validation metrics
- ONNX export
- calibration export
- MXQ handoff metadata

### Artifact Contract

Final artifacts should mirror the repo standard:

- best checkpoint `.pth`
- exported `.onnx`
- `.mxq` handoff target path
- calibration directory built from training-derived samples
- `summary.json`

Recommended output root:

- `runs/<started_day>/<run_name>/`

## NPU and Operator Constraints

### Allowed Path for v1

The intended forward path should stay inside this operator family as much as possible:

- `Conv2d`
- `TransposeConv2d`
- `LeakyReLU`
- `InstanceNormalization`
- residual `Add`

### Avoid Path for v1

Do not introduce the following in the primary implementation:

- attention gates using `Multiply`
- supervised attention modules
- `Softmax`
- `MatMul`
- concat-based skip fusion in the forward path unless export proves safe
- dynamic reshape-heavy blocks
- resize paths that introduce extra fallback ops when a transposed-conv path suffices

### Important Notes

- `InstanceNormalization` is acceptable in the local MLA notes.
- Full HiNet is reduced mainly to cut concat/fusion complexity and latency risk, **not** because HIN itself is disallowed.
- Export audit for this model should confirm the forward graph does **not** contain unexpected attention-style ops such as `Softmax`, `MatMul`, or `Multiply`.

## Training Plan

### Optimizer and Loss

Training defaults for v1:

- Loss: `L1` only
- Optimizer: `AdamW`
- EMA: enabled
- Mixed precision: enabled if it remains consistent with the repo’s baseline pipeline behavior

Out of scope for v1:

- perceptual loss
- adversarial loss
- teacher distillation
- multi-stage objective scheduling

### Augmentation and Sampling

Training preprocessing must match the stronger baseline path already established in the repo:

- joint random `224x224` crop
- horizontal flip
- vertical flip
- `k * 90` rotation

Required behavior:

- all train transforms must be synchronized between LR and HR
- augmentation seeding must be epoch-aware
- the same sample must not receive the same crop forever across epochs

### Training Defaults

- `train_patch_size = 224`
- `eval_size = 256`
- Global residual prediction must remain enabled
- Initial experiments should start with the default `channels = 32`
- A larger `channels = 48` follow-up may be used later as a controlled capacity ablation, but it is not the primary v1 target

## Evaluation Plan

Validation must run on the parity-checked validation path only.

Required reported metrics:

- `val_psnr`
- `input_psnr`
- `delta_psnr`

Required validation behavior:

- include `val_preview`
- log the validation slice identity clearly
- enforce expected baseline-slice parity behavior in the spirit of [`docs/notes/span_modal_psnr_diagnosis_20260421.md`](../notes/span_modal_psnr_diagnosis_20260421.md)

Interpretation rule:

- bounded or smoke results must not be presented as full-run numbers
- the implementation should require clear distinction between smoke and full validation modes

## Export and Submission Path

### ONNX

The notebook flow must include ONNX export for the best checkpoint.

Required checks:

- export succeeds
- model checker passes
- parity check runs on CPU
- op audit confirms the forward graph stays inside the intended operator family

### Calibration

Calibration artifacts must come from **training-derived data** only.

Requirements:

- write calibration samples under the run export directory
- keep calibration generation reproducible
- record enough metadata to show that calibration inputs came from train pairs, not validation data

### MXQ Handoff

The notebook must produce a clear handoff path for:

- checkpoint path
- ONNX path
- calibration directory
- intended MXQ output path

The v1 document does not require local compilation of MXQ, but it does require an explicit handoff path consistent with the repo’s existing Lab 3 export workflow.

## Acceptance Tests

The implementation is not complete until the following tests pass.

### 1. Model Shape Tests

- Input `1x3x256x256` returns `1x3x256x256`
- Intermediate encoder and decoder channel counts match `HiNetLiteConfig`
- Skip-add fusion shapes are valid at both decoder levels

### 2. Block Correctness Tests

- `HINResidualBlock` preserves tensor shape
- Only half the channels are instance-normalized
- Residual path runs on random input without NaNs

### 3. Data-Path Tests

- Train and validation loaders use canonical `(lr, hr, name)` pairing
- Full-run audit passes before loader creation
- Train augmentation is synchronized between LR and HR
- Train augmentation changes across epochs for the same index

### 4. Training Smoke Test

- A bounded run completes end to end on a small slice
- Validation metrics include `input_psnr` and `delta_psnr`
- Model output beats or at least clearly differs from identity before longer runs are attempted

### 5. Export Tests

- ONNX export succeeds
- ONNX parity check passes on CPU
- Operator audit shows no unexpected attention-style ops: `Softmax`, `MatMul`, `Multiply`
- Calibration bundle is written from train-derived samples

## Assumptions and Defaults

- “docs repo” means this repository’s `docs/` directory.
- This plan is for **HiNet-lite only**, with only brief context about why full HiNet was simplified.
- The first implementation target is a **single-stage** model optimized for repo fit and operator discipline, not paper fidelity.
- `TransposeConv2d` is the default upsampling choice for v1.
- Skip fusion is **add-based**, not concat-based, unless later export evidence justifies a change.
- v1 uses `L1` only; perceptual, adversarial, and teacher-distillation losses are explicitly out of scope.
