# SRNet Current Work Review

**Date:** 2026-05-02  
**Notebook:** [experiments/SRNet NPU v1/lab3_srnet_npu_v1_modal.ipynb](../../experiments/SRNet%20NPU%20v1/lab3_srnet_npu_v1_modal.ipynb)  
**Plan:** [docs/model_plans/SRNet-1.md](../model_plans/SRNet-1.md)

---

## Goal

Train a **Mobilint MLA-100-safe** super-resolution model (256×256×3 → 256×256×3) that surpasses FSRCNN's best validation PSNR while passing a strict ONNX operator audit against the MLA PDF whitelist.

---

## PSNR Reference Points

| Model | Val PSNR | Notes |
|---|---|---|
| FSRCNN 96/40/m8 @ 6000 epochs | 24.0841 | Best epoch: 5993 |
| FSRCNN 96/40/m8 @ 8000 epochs | **24.1335** | Best epoch: 7994 |
| HiNetLite NPU v6 | 23.9301 | |
| SRNet recovery best (warm start source) | 24.0179 | Prior SRNet run |
| SRNet initial plateau | 23.1359 | Early SRNet baseline |

**SRNet targets:**
- Minimum: `> 24.0179` (beat prior SRNet recovery)
- Goal: `>= 24.0634` (beat original FSRCNN reference)
- Stretch: `>= 24.1335` (beat FSRCNN @ 8000 epochs)

---

## Why SRNet Can't Just Use FSRCNN

FSRCNN uses a global residual `output = x + delta`. The `Add` op is listed in the Mobilint MLA-100 PDF as a **fallback-risk operator** — it may work in simulation but is not guaranteed on-device. The strict audit (`PDF_ALLOWED_ONNX_OPS`) accepts only `Conv`, `ConvTranspose`, and `LeakyRelu`. Any `Add`, `Concat`, `Mul`, `Sigmoid`, `Resize`, or similar ops cause an automatic fail.

SRNet must achieve competitive quality **without any residual addition at inference time.**

---

## Architecture: SRNetMLAStrictPreDec2Bottleneck

The model is a pure-convolutional U-Net. No skip adds, no global residual, no attention.

```
Input: 1×3×256×256
│
├─ Stem: Conv3x3(3→32) + LeakyReLU                           [256×256]
│
├─ Encoder 1: 3× ResidualCleanBlock(32, k=3)                 [256×256]
├─ Down 1:    Conv3x3 stride2(32→64) + LeakyReLU             [128×128]
│
├─ Encoder 2: 3× ResidualCleanBlock(64, k=5)                 [128×128]
├─ Down 2:    Conv3x3 stride2(64→128) + LeakyReLU            [ 64×64]
│
├─ Bottleneck: 5× ResidualCleanBlock(128, kernels 3,5,3,5,3) [ 64×64]
│
├─ Up 1:       ConvTranspose2d(128→64, stride2) + LeakyReLU  [128×128]
├─ Decoder 1: 3× ResidualCleanBlock(64, k=5)                 [128×128]
│
├─ Up 2:       ConvTranspose2d(64→32, stride2) + LeakyReLU   [256×256]
│
├─ Pre-Dec2 Bottleneck (new):                                 [256×256]
│   ├─ Conv3x3 stride2(32→48) + LeakyReLU                   [128×128]
│   ├─ 3× ResidualCleanBlock(48, kernels 3,5,3)              [128×128]
│   └─ ConvTranspose2d(48→32, stride2) + LeakyReLU           [256×256]
│
├─ Tail Head: Conv3x3(32→32) + LeakyReLU                     [256×256]
└─ Tail:      Conv3x3(32→3)                                  [256×256]

Output: 1×3×256×256  (direct image, no residual add)
```

**ResidualCleanBlock** (no skip):
```
Conv2d → LeakyReLU → Conv2d → LeakyReLU
```
The "residual" in the name refers only to the overall model's role (residual learning in loss space), not a skip connection inside the block.

**What the Pre-Dec2 Bottleneck adds:**  
After the second upsampler returns to 256×256 resolution, the signal passes through an extra stride-2 sub-network that compresses to 128×128, applies 3 residual-style blocks, then expands back. This gives the model more capacity to refine spatial detail at half-resolution before the final output head, without adding parameters at full 256×256 resolution.

---

## Training Setup

| Parameter | Value |
|---|---|
| GPU | Modal L40S |
| Timeout | 1440 min (24 h) |
| Epochs | 460 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| LR schedule | Cosine decay |
| Min LR | 5e-6 |
| Warmup epochs | 5 |
| Weight decay | 5e-5 |
| Grad clip norm | 0.5 |
| AMP | Yes (float16 on CUDA) |
| EMA decay | 0.999 |
| Early stop patience | 35 epochs |

**Loss function:**
```
pred_res    = pred - lr
target_res  = hr - lr
teacher_res = teacher_pred - lr

loss_mse  = MSE(pred_res, target_res)
loss_l1   = L1(pred_res, target_res)
loss_dist = MSE(pred_res, teacher_res.detach())

loss = 0.8 × loss_mse + 0.2 × loss_l1 + 0.05 × loss_dist
```
Loss is computed in **residual space** (prediction minus LR input), which keeps gradient magnitudes stable and aligns teacher and student targets.

---

## Teacher & Warm Start

### FSRCNN Teacher
- Model: `FSRCNNResidualNoUpscale` (96 channels, 40 shrink, 8 mapping layers)
- Role: Training-only distillation target — never exported or submitted
- Checkpoint resolution order:
  1. `runs/013420_2904_fsrcnn_residual_96_40_m8_modal_resume5750_to6000_lr300/checkpoints/best.pth`
  2. `runs/021424_2904_fsrcnn_residual_96_40_m8_modal_resume5750_to6000_lr300/checkpoints/best.pth`
- If neither exists: distillation is silently disabled and recorded in `report.json`

### Warm Start
- Source: `runs/2026-05-01/lab3_srnet_dwg4_v1_recovery_l3_20260501_1651/checkpoints/best.pth`
- Strategy: Load only keys where name and shape match; skip all mismatches
- This initialises the encoder/decoder from the prior SRNet recovery run (PSNR 24.0179), giving the new bottleneck layers a running start instead of training from scratch

---

## MLA-100 Operator Constraints

The ONNX graph is audited post-export against the MLA PDF whitelist:

| Status | Ops |
|---|---|
| Allowed | `Conv`, `ConvTranspose`, `LeakyRelu` |
| Forbidden (auto-fail) | `Add`, `Concat`, `Clip`, `Mul`, `Sub`, `Relu`, `Resize`, `MatMul`, `Softmax`, `Sigmoid`, `Transpose`, `Tanh` |

The audit sets `onnx_summary.pdf_allowed_ops_only = true/false` in the run report. A single forbidden op is a hard failure regardless of PSNR.

---

## Export Chain

```
Training (Modal L40S)
    └─ best.pth  (EMA weights preferred, model weights fallback)
         └─ ONNX export  (opset 17, input 1×3×256×256)
              ├─ onnx.checker pass
              ├─ ORT parity check  (max_diff < 1e-3)
              ├─ PDF operator audit
              └─ calibration/  (128 training-derived LR images, deterministic seed)
                   └─ convert_srnet_mla_strict_predec2_v1_mxq.py
                        └─ srnet_mla_strict_predec2_v1.mxq
```

---

## Dataset

| Split | Pairs |
|---|---|
| Train (5 sub-splits) | ~2,217 |
| Val | ~110 |

All images are 256×256 RGB. Calibration images are sampled from the training split only (never validation), using a fixed seed for reproducibility.

---

## Acceptance Criteria

1. `onnx_summary.pdf_allowed_ops_only == true` (hard requirement)
2. `best_val_psnr > 24.0179` (beat prior SRNet recovery)
3. `best_val_psnr >= 24.0634` (primary goal — beat FSRCNN reference)
4. If PSNR regresses below 24.0179 but the audit passes, the run is classified as a **latency-oriented architectural experiment**, not the new primary SRNet

---

## What Has Been Tried (SRNet Evolution)

| Stage | Architecture | Outcome |
|---|---|---|
| Initial SRNet | Add-based U-Net with global skip | Failed MLA audit (`Add` in exported graph) |
| SRNet recovery | Add-free U-Net (no pre-dec2) | 24.0179 PSNR, audit passes |
| **SRNet Pre-Dec2** | Add-free U-Net + pre-dec2 bottleneck | **Current run** |

---

## Open Questions

- Does the pre-dec2 bottleneck actually recover the capacity lost by removing all skip adds, or does it plateau similarly to the recovery run?
- At 460 epochs with a warm start, is the training budget long enough to see meaningful improvement over 24.0179?
- The FSRCNN @ 8000 epochs now sits at 24.1335 — should the SRNet target be updated to reflect this new ceiling?
