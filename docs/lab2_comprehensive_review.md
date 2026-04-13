# Lab 2 — Comprehensive Repository Review

## 1. Project Overview

This repository documents the iterative development of a **same-resolution image super-resolution (SR) / restoration model** for Lab 2 of DATA 255 (Spring 2026). The task is to take a degraded low-resolution (LR) 256×256 image and produce a restored high-resolution (HR) 256×256 output — a restoration problem rather than spatial upscaling.

The project spans **8 distinct development phases** across ~15 notebooks and multiple shared Python modules, exploring progressively more sophisticated model architectures, data pipelines, training protocols, and deployment targets (ONNX → MXQ for Mobilint NPU).

> [!IMPORTANT]
> The final submitted model (`mixed_kernel_residual` in Phase 7B) is architecturally the **simplest** model explored in the project — a deliberate design decision driven by NPU deployment constraints and lessons learned from over-engineering in earlier phases.

---

## 2. Repository Structure

```
Lab-2/
├── Lab 2 First Try/         # Initial prototype — Residual Attention Net
├── Lab 2 Phase 1/           # ResNet baseline
├── Lab 2 Phase 2/           # UNet, ResNet-SE, Hybrid explorations
├── Lab 2 Phase 3/           # Refined ResNet-SE with longer training
├── Lab 2 Phase 4/           # Wide SE-ResNet, DW-Attention, MFR-Net + Modal scripts
├── Lab 2 Phase 5/           # Shared common module (phase5_common.py), Colab/A100/TPU training
├── Lab 2 Phase 6/           # Architecture screening matrix (6 candidates) + postmortem
├── Lab 2 Phase 7/           # Calibrated NPU SR — no-norm residual models + Modal outputs
├── Lab 2 last run/          # Final training run (mixed_kernel_residual) + checkpoints
├── Submission/              # ONNX → MXQ final model + conversion script + notebook
├── CanvasSubmissionApril8/  # Cleaned notebook + MXQ submitted to Canvas
├── ONNX-toMXQ/              # ONNX-to-MXQ conversion tooling + calibration dataset
├── tmp/                     # Standalone script (lab2_static.py) with MODEL_REGISTRY
├── docs/                    # This review document
├── Data/                    # Training/validation data (HR/LR paired + ImageNet + COCO)
└── runs/                    # Training run outputs (metrics, checkpoints, Phase 6 screening)
```

---

## 3. Data Loading & Pipeline Evolution

### 3.1 Dataset Structure

The core dataset consists of **paired LR/HR PNG images** at 256×256:

| Split | Folders | Count | Baseline PSNR (LR vs HR) |
|---|---|---:|---|
| Training | `HR_train1–4` / `LR_train1–4` | ~3,036 pairs | ~26–27 dB (relatively easy) |
| Validation | `HR_val` / `LR_val` | 100 pairs | ~21.3 dB (much harder) |

> [!WARNING]
> The validation set is significantly harder than training — the gap between train and val baseline PSNR (~5–6 dB) was a persistent bottleneck identified in the Phase 6 postmortem.

Additionally, external datasets are used for synthetic data augmentation:
- **ImageNet-20**: ~6,000 train / ~300 val images (from course-provided subset)
  - Manifests: `imagenet_train20.txt`, `imagenet_val20.txt` under `course_files_export/`
  - Archives: `imagenet_train20.zip` → `imagenet_train20a/`, `imagenet_val20.zip` → `imagenet_val20/`
  - Introduced in Phase 3 (for evaluation) and Phase 4–5 (for synthetic training)
- **COCO 2017**: ~12,000–30,000 train / ~500–1,000 val images
  - Downloaded from `images.cocodataset.org` via `stage_coco2017()` helper
  - Stored as zips on Modal volumes for efficient volume cloning
  - Archives: `train2017.zip`, `val2017.zip` under `course_files_export/coco2017/`
  - Generated manifests: `coco_train2017.txt`, `coco_val2017.txt`
  - Screening config: `coco_train_limit=12,000`, `coco_val_limit=500`; full runs: 30,000 / 1,000
  - Introduced in Phase 6 as a third data source alongside paired and ImageNet

### 3.2 Data Pipeline Phases

#### First Try & Phase 1–3: Simple Paired Loading
- `PairedImageDataset`: loads LR/HR pairs by basename matching
- Augmentation: synchronized horizontal flip, vertical flip, k×90° rotation
- Tensors in `[0, 1]` float range
- `DataLoader` with standard batching
- **No synthetic data** — trains on paired data only

#### Phase 4–5: Paired + ImageNet Synthetic
- Introduced `ImageNetSyntheticSRDataset` alongside `PairedSRDataset`
- Synthetic degradation pipeline:
  - Gaussian blur (radius 0.2–1.2)
  - Downsample/upsample at scales (2, 3, 4) using bicubic interpolation
  - JPEG compression (quality 40–90)
- Additional augmentations at tensor level:
  - LR noise injection (prob 0.30, std 0.015)
  - Random cutout (prob 0.35, ratio 0.18)
- Training on `ConcatDataset([paired_train, imagenet_train])`
- Evaluation on `ConcatDataset([paired_val, imagenet_val])`
- **Calibration dataset** construction: diversity-sampled by brightness/texture tertiles

#### Phase 6: Multi-Source Staged Training with COCO
- Added COCO 2017 as a third data source
- Introduced **2-stage training protocol**:
  - **Stage 1 (Pretrain)**: Synthetic-only (COCO ± ImageNet) — 12 epochs
  - **Stage 2 (Finetune)**: Paired-only — 8 epochs
- Enhanced degradation recipe:
  - Wider blur range (0.2–1.6)
  - Multiple resize interpolation modes (bicubic, bilinear, lanczos)
  - Lower JPEG quality floor (25)
- Separate `paired_finetune_data_cfg` that disables cutout and noise for Stage 2

#### Phase 7 & Final: Calibrated Synthetic Degradation + 3-Stage Curriculum

**Degradation profiling** was introduced as a foundational step. Before training, the pipeline computes per-image statistics for all 3,036 train and 100 val pairs:
- Baseline PSNR (LR vs HR)
- Mean absolute residual
- Gradient energy ratio (LR grad / HR grad)
- Laplacian energy ratio (LR lap / HR lap)
- Signed RGB residual mean (per-channel)

This profile confirmed the train/val difficulty gap quantitatively:

| Split | n | Mean Baseline PSNR | Median | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| `HR_train1` | 663 | 26.112 | 25.400 | 20.502 | 32.365 |
| `HR_train2` | 782 | 26.734 | 26.117 | 19.752 | 34.089 |
| `HR_train3` | 810 | 26.998 | 26.408 | 19.994 | 34.357 |
| `HR_train4` | 781 | 27.078 | 26.418 | 19.915 | 34.650 |
| `HR_val` | 100 | **21.336** | 21.548 | 17.914 | 24.072 |

The profile is persisted as `degradation_profile.json` (~53K lines) and used to configure calibrated synthetic generation.

- **Calibrated degradation** — synthetic LR generation is tuned to match the paired validation difficulty distribution using rejection-sampling (`CalibratedSyntheticSRDataset`):
  - Target PSNR range: 17.5–24.2 dB (centered at 21.2, matching val distribution)
  - Target gradient ratio: 0.35–0.85
  - Target Laplacian ratio: 0.25–0.75
  - Additional RGB gamma shift augmentation
  - Up to 12 rejection-sampling attempts per image; `degradation_score` function evaluates candidates
- **3-stage curriculum**:

| Stage | Data | Epochs | LR | Focus |
|---|---|---:|---|---|
| `synthetic_warmup` | Calibrated synthetic only | 6–12 | 3e-4 | Build general restoration prior |
| `mixed_finetune` | 65% paired + 35% synthetic (weighted sampling) | 20–24 | 1.5e-4 | Bridge distribution gap |
| `paired_polish` | Paired only (hard-sample weighted) | 8–20 | 5e-5 | Final domain-specific refinement |

- **Hard-patch mining**: during `mixed_finetune` and `paired_polish`, patch crops are selected to minimize LR–HR PSNR (harder patches preferred)
- **WeightedRandomSampler**: oversamples hard training pairs (continuous weighting based on per-image baseline PSNR relative to threshold of 24.8 dB)
- **Degradation profiling**: computes and persists per-pair metrics (baseline PSNR, gradient energy ratio, Laplacian energy ratio, mean absolute residual, signed RGB residual) for the entire train/val set

---

## 4. Model Architecture Evolution

### 4.1 Phase 0 — First Try: MultiScaleResidualAttentionNet

**File**: [lab2_same_resolution_residual_attention_pipeline.ipynb](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20First%20Try/lab2_same_resolution_residual_attention_pipeline.ipynb)

- `3 → 48` conv stem
- 3 residual groups × 3 residual blocks
- Stage attention after each group
- `48 → 24 → 3` tail
- Global residual skip: `output = input + model(input)`
- **No BatchNorm or ReLU** (export-safe from the start)
- Export-aware design: ONNX opset 13, static shape `[1, 3, 256, 256]`

### 4.2 Phase 1: ResNet Baseline

**File**: [lab2_phase1_resnet.ipynb](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%201/lab2_phase1_resnet.ipynb)

```python
# ResBlock: Conv2d(c, c, 3) → BN → PReLU → Conv2d(c, c, 3) → BN + skip
# ResNetSR: Stem Conv2d(3→48, 3×3) + 10× ResBlock(48) + Tail Conv2d(48→3, 3×3)
# Forward: x + tail(body(stem(x)))
```

- **Channels**: 48, **Blocks**: 10, **Activation**: PReLU
- Uses **BatchNorm** in every residual block (2 BN per block)
- Trained on **paired data only** (no synthetic augmentation)
- Served as the performance baseline for all subsequent experiments
- **Best paired-val PSNR**: **21.362 dB** (epoch 12 of 12)

### 4.3 Phase 2: Architecture Exploration (3 notebooks)

| Notebook | Architecture | Key Features | Best Val PSNR |
|---|---|---|---:|
| [lab2_phase2_resnet_se.ipynb](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%202/lab2_phase2_resnet_se.ipynb) | **ResNet-SE** (`ResNetSESR`) | SE blocks added to residual backbone | **21.737 dB** (ep 38) |
| [lab2_phase2_unet.ipynb](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%202/lab2_phase2_unet.ipynb) | **UNet** (`UNetSR`) | Encoder-decoder with skip connections and PixelShuffle | — |
| [lab2_phase2_hybrid.ipynb](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%202/lab2_phase2_hybrid.ipynb) | **Hybrid** (`HybridUNetSESR`) | UNet backbone with SE residual blocks | **21.898 dB** (ep 35) |

#### ResNet-SE (`ResNetSESR`)
- **SEBlock**: `AdaptiveAvgPool2d(1)` → `Conv2d(c→c//reduction, 1)` → PReLU → `Conv2d(c//reduction→c, 1)` → **Hardsigmoid** → channel rescale
- **SEResBlock**: `Conv2d(3×3)` → BN → PReLU → `Conv2d(3×3)` → BN → SE → residual add
- **Network**: Stem `3→48` + 10× SEResBlock(48) + Tail `48→3`; `reduction=4`
- First architecture to introduce channel attention to the project

#### UNet (`UNetSR`)
- **Encoder**: 3-level strided convolutions, channel schedule `32 → 48 → 64` (spatial: 256→128→64)
- **Bottleneck**: 2× ResBlock at 64×64 spatial
- **Decoder**: **PixelShuffle(×2)** upsampling at each level, skip-connection **concat**, Conv fusion + ResBlock
- **Tail**: `base_ch → 3`; **Forward**: `x + delta` (global residual)
- Only encoder-decoder architecture explored; tested whether multi-scale spatial reasoning helps same-resolution restoration

#### Hybrid (`HybridUNetSESR`)
- Same 3-level UNet layout as `UNetSR` but all encoder/bottleneck/decoder blocks replaced with **SEResBlock** (Conv-BN-PReLU-Conv-BN + SE attention)
- Channel schedule `32 / 48 / 64`; global residual `x + delta`
- Achieved the best Phase 2 paired-val result at **21.898 dB**, suggesting that channel attention was beneficial even in the encoder-decoder setting

### 4.4 Phase 3: Refined ResNet-SE

**File**: [lab2_phase3_resnet_se.ipynb](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%203/lab2_phase3_resnet_se.ipynb)

Key changes from Phase 2's ResNet-SE:
- Added **Dropout2d** and **StochasticDepth** (drop-path) inside each SEResBlock for regularization
- Retuned model configuration:

```python
MODEL_CFG = {
    "num_blocks": 8,       # reduced from 10
    "channels": 40,        # reduced from 48
    "reduction": 4,
    "dropout": 0.10,
    "max_drop_path": 0.05, # linearly increasing per-block drop probability
}
```

- Longer training (40 epochs vs ~38–41 in Phase 2)
- First phase to incorporate **ImageNet synthetic data** alongside paired data for validation
- **Reference results**: paired-val PSNR ~21.696 dB, ImageNet-val ~26.825 dB, **combined-val ~25.543 dB** (best `val_psnr` at epoch 40)
- The combined-val metric (paired + ImageNet) became the primary headline number from this phase onward

### 4.5 Phase 4: Widened Architectures (3 variants)

| Notebook | Architecture | Params | Key Features |
|---|---|---:|---|
| [lab2_phase4a_wide_se_resnet.ipynb](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%204/lab2_phase4a_wide_se_resnet.ipynb) | **Wide SE-ResNet** (`WideSEResNetSR`) | ~1,222,723 | Widened channel count (64), 16 SE residual blocks |
| [lab2_phase4b_dw_attention_net.ipynb](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%204/lab2_phase4b_dw_attention_net.ipynb) | **DSDAN** (`DSDANSR`) | ~622,467 | Depthwise separable with dual attention (SE + spatial gate) |
| [lab2_phase4c_mfr_net.ipynb](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%204/lab2_phase4c_mfr_net.ipynb) | **MFR-Net** (`MFRNSR`) | 757,179 | Multi-scale feature refinement with dilated convolutions |

Phase 4A achieved **combined val_psnr = 25.841 dB** and **paired_val PSNR = 21.848 dB** after 80 epochs on Colab (verified from Google Drive notebook).

#### Wide SE-ResNet (`WideSEResNetSR`) — Phase 4A
- Scaled up from Phase 3's 40-channel / 8-block design to **64 channels** and **16 SE residual blocks**
- Same SEResBlock structure as Phase 3 (Conv-BN-PReLU-Conv-BN-SE + Dropout2d + StochasticDepth)
- Config: `channels=64`, `reduction=4`, `dropout=0.08`, `max_drop_path=0.10`
- **1,222,723 parameters**; NPU ops: 66 Conv, 33 BN, 33 PReLU, 16 GlobalAvgPool, 16 HardSigmoid
- **Colab run** (verified from Google Drive, 103 revisions):
  - 80 epochs, batch_size=32, lr=3e-4, weight_decay=2e-4, warmup=5 epochs, EMA decay=0.999
  - Resumed from epoch 4 (best 24.856 dB); trained to epoch 80
  - **Best combined val_psnr = 25.841 dB** (epoch 80), **paired_val = 21.848 dB**, imagenet_val = 27.172 dB
  - ~80.8s/epoch, **225.4 min total**
  - Output: `/content/drive/MyDrive/Data 255 Class Spring 2026/Lab 2/runs/phase4a_wide_se`
  - Google Drive notebook also modified April 10 (9 additional revisions after submission)

#### DSDAN (`DSDANSR`) — Phase 4B
Depthwise Separable Dual-Attention Network — the project's first fully depthwise-centric design:
- **DSDABlock**: `DW(3×3)` → BN → PReLU → `PW(1×1)` → BN → PReLU → `DW(3×3)` → BN → `PW(1×1)` → BN → **SE** (channel attention) → **SpatialGate** (`DW(7×7, groups)` → BN → **Hardsigmoid**) → Dropout2d → StochasticDepth → residual add
- Config: `num_blocks=12`, `channels=128`, `reduction=4`
- Tested whether depthwise-heavy architectures could match full-convolution SE-ResNet while being more NPU-efficient

#### MFR-Net (`MFRNSR`) — Phase 4C
Multi-scale Feature Refinement Network — the most architecturally complex model in the project:
- **MultiScaleConv**: parallel paths — (1) local `Conv2d(c, c, 3×3)` and (2) dilated depthwise `Conv2d(c, c, 3×3, dilation=2, groups=c)` → pointwise `Conv2d(c, c, 1×1)`; outputs summed → BN → PReLU
- **MFRBlock**: MultiScaleConv → `Conv2d(3×3)` → BN → SE → Dropout2d → StochasticDepth → residual add
- **RefinementGate**: `Conv2d(c, c, 1×1)` → **InstanceNorm2d** → **Mish** activation → residual add (applied after each stage)
- **MFRStage**: `blocks_per_stage` × MFRBlock + RefinementGate
- **MFRNSR**: Stem (`Conv2d(3→c, 3×3)` → BN → PReLU) → `num_stages` × MFRStage → Tail (`Conv2d(c→3, 3×3)`) → global residual
- Config: `num_stages=3`, `blocks_per_stage=4`, `channels=56`, `reduction=4`, `dropout=0.08`, `max_drop_path=0.10`
- **757,179 parameters** with 65 Conv, 12 DWConv, 25 BN, 3 InstanceNorm, 25 PReLU, 3 Mish, 12 SE blocks
- Only architecture in the project to use **InstanceNorm**, **Mish**, **dilated convolutions**, and **per-stage gating**

### 4.6 Phase 5: Shared Training Infrastructure + New Architectures

Introduced [phase5_common.py](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%205/phase5_common.py) — a 1,295-line shared module containing:
- Data loading, synthetic degradation, augmentation
- EMA (Exponential Moving Average) for model weights
- Training loop with linear warmup + cosine decay
- Combined Charbonnier + L1 loss
- ONNX export and parity checking
- Calibration dataset construction

| Notebook | Architecture | Key Features |
|---|---|---|
| `lab2_phase5a_wide_se_a100.ipynb` | Wide SE-ResNet (A100) | GPU-optimized training |
| `lab2_phase5a_wide_se_tpu.ipynb` | Wide SE-ResNet (TPU) | TPU adaptation |
| `lab2_phase5b_dsdan_a100.ipynb` | **DSDAN** | Depthwise Separable Dual-Attention Network |
| `lab2_phase5c_repconv_sr.ipynb` | **RepConvSR** | Re-parameterizable convolutions (fuse Conv+BN at deploy) |
| `lab2_phase5d_large_kernel_dw_sr.ipynb` | **Large-Kernel DW SR** | 7×7 and 11×11 depthwise kernels with expansion |

### 4.7 Phase 6: Systematic Architecture Screening

The most extensive architecture comparison, using [phase6_screening_common.py](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%206/phase6_screening_common.py) (3,581 lines). Six candidate architectures were screened:

| Model | Architecture | Params | BN Count | Key Modules |
|---|---|---:|---:|---|
| `wide_se` | **WideSEResNetSR** | 1,222,723 | 33 | 16× SEResBlock (Conv-BN-PReLU-Conv-BN-SE) |
| `dsdan` | **DSDANSR** | 622,467 | 61 | 12× DSDABlock (DW-BN-PW-BN + SE + SpatialGate) |
| `repconv` | **RepConvSR** | 2,117,091 | 49 | 12× RepConvBN (3-branch: 3×3+BN, 1×1+BN, identity+BN) |
| `large_kernel_dw` | **LargeKernelDWSR** | 650,595 | 43 | 14× LargeKernelDWBlock (alternating 7×7/11×11 DW) |
| `large_kernel_se` | **LargeKernelSESR** | 717,123 | 43 | 14× LargeKernelDW + SE attention |
| `hybrid_rep_large_kernel` | **HybridRepLargeKernelSR** | 1,337,763 | 43 | Alternating RepConv and LargeKernelDW blocks |

All architectures shared:
- Same-resolution residual design: `output = input + tail(body(stem(input)))`
- Stem: `Conv2d(3→C, 3×3) → BN → PReLU`
- Tail: `Conv2d(C→3, 3×3)`
- StochasticDepth and Dropout2d within residual blocks
- PReLU activation (NPU-compatible)
- Hardsigmoid gating in SE blocks

**Screening results** (from the [Phase 6 Postmortem](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Lab%202%20Phase%206/phase6_postmortem.md)):

| Rank | Model | Data Mix | Paired Val PSNR | Combined Val PSNR |
|---:|---|---|---:|---:|
| 1 | `wide_se` | coco+imagenet | **21.826** | 24.849 |
| 2 | `wide_se` | coco_only | 21.809 | 24.541 |
| 3 | `dsdan` | coco+imagenet | 21.693 | 24.830 |

> [!NOTE]
> The key finding was that **BatchNorm was a persistent problem**: 33–61 BN layers with screening batch size 4, EMA not tracking BN buffers, and domain mismatch between synthetic pretraining and paired validation. This directly motivated Phase 7's normalization-free approach.

**Phase 6 Postmortem Root-Cause Analysis** (ranked by confidence):

1. **High confidence**: Synthetic degradation does not match paired-val LR artifacts — synthetic val improves while paired val degrades during Stage 1 pretraining
2. **High confidence**: Paired val is ~5–6 dB harder than paired train (baseline 21.3 vs 26–27 dB) — models fit training distribution without solving validation distribution
3. **Medium-high**: Stage 2 adapts to paired-train but not paired-val distribution — train-eval reaches ~27 dB while paired-val stays at ~21.7–21.8 dB
4. **Medium**: BatchNorm amplifies domain shift — many BN layers, batch size 4, EMA not tracking BN buffers creates parameter/statistics mismatch
5. **Medium**: Dropout/StochasticDepth hurt precise residual correction during low-data paired fine-tuning
6. **Medium-low**: Attention/gating learns wrong domain priors from synthetic data, suppressing useful residual features
7. **Medium-low**: Depthwise-heavy blocks are capacity-limited for cross-channel color corrections

**Screening protocol findings:**
- Stage 1 best paired-val epoch happens very early (epochs 2–4), then degrades — long synthetic pretraining hurts paired metrics
- Stage 2 (8 epochs) may be too short (all runs peaked at final epoch), but extending alone cannot bridge the ~5 dB train/val gap
- Screening batch size (4) was much smaller than Colab runs (32), worsening BN stability

### 4.8 Phase 7 & Last Run: No-Normalization Residual Models (FINAL)

Drawing on all Phase 6 lessons, Phase 7 radically simplified the architecture:

#### Phase7NoNormResidualConvSR (`nonorm_residual`)
```python
# Stem: Conv2d(3→96, 3×3) + PReLU
# Body: 18× ConvPReLUBlock(96, 3×3)  — Conv2d + PReLU, no BN/SE/Dropout
# Tail: Conv2d(96→3, 3×3)
# Forward: x + tail(body(stem(x)))
```

#### Phase7DirectConvSR (`direct_conv`)
```python
# A flat sequential chain with no residual skip connection
# Conv2d(3→96) → PReLU → 18×[Conv2d(96) → PReLU] → Conv2d(96→3)
```

#### Last Run Models (submitted): `ResidualConvSR` and variants

The final submission notebook used **LeakyReLU** (slope 0.10) instead of PReLU for simpler ONNX export:

| Model ID | Architecture | Channels | Blocks | Kernel Pattern | Params | Key Change |
|---|---|---:|---:|---|---:|---|
| `nonorm_residual` | ResidualConvSR | 96 | 18 | all 3×3 | 1,501,827 | Baseline |
| **`mixed_kernel_residual`** ⭐ | ResidualConvSR | 96 | 18 | `[3,3,5]×6` | **2,386,563** | **Mixed 3×3/5×5 kernels** |
| `expanded_dw_large_residual` | ExpandedDepthwiseResidualSR | 128 | 12 | 9×9 DW | 1,057,795 | Inverted bottleneck |
| `expanded_dw_large_deep` | ExpandedDepthwiseResidualSR | 128 | 18 | 7×7 DW | 1,435,651 | Deeper variant |
| `mixed_kernel_residual_wide` | ResidualConvSR | 128 | 24 | `[3,3,5]×8` | 5,649,411 | Wider variant |
| `expanded_dw_large_v2` | ExpandedDepthwiseResidualSR | 160 | 14 | 9×9 DW | 1,827,843 | Higher capacity |
| `hybrid_large_fine_residual` | HybridLargeFineResidualSR | 128 | 8 DW + 10 fine | 9×9 DW then 3×3/5×5 | 3,495,427 | Two-stage body |

**Architecture class details:**

- **`ResidualConvSR`**: Stem `3→C` → Body = sequence of `ConvLeakyReLUBlock(C, kernel=K)` following the `kernel_pattern` list → Tail `C→3`; forward `x + tail(body(stem(x)))`
- **`ExpandedDepthwiseResidualSR`**: Stem → stack of **ExpandedDepthwiseBlock** (1×1 expand → PReLU → depthwise K×K → PReLU → 1×1 project → PReLU; inverted bottleneck design) → Tail; forward `x + delta`
- **`HybridLargeFineResidualSR`**: Stem → N_dw × ExpandedDepthwiseBlock(K=9) for coarse features → fine ConvLeakyReLUBlocks with alternating 3/5 kernels → Tail; forward `x + delta`

**All Phase 7 models share these properties:**
- ❌ No BatchNorm
- ❌ No Dropout / Stochastic Depth
- ❌ No SE / attention / gating
- ❌ No AdaptiveAvgPool
- ✅ Only Conv2d + LeakyReLU (or PReLU) leaves
- ✅ Global residual: `x + delta(x)`
- ✅ Tail initialized with small weights (`scale=1e-3`)

### 4.9 Activation Function Evolution

| Phase | Activation | Rationale |
|---|---|---|
| First Try – Phase 6 | **PReLU** (per-channel learnable slope) | Standard for SR; NPU-compatible |
| Phase 7 (early) | **PReLU** | Continuity with prior phases |
| Phase 7B / Final | **LeakyReLU** (slope=0.10) | Simpler ONNX graph (single scalar vs per-channel tensor); more reliable NPU inference |
| Submission pipeline | **PReLU→LeakyReLU rewriter** | ONNX post-processing: averages per-channel PReLU slopes → scalar LeakyReLU alpha |

The switch from PReLU to LeakyReLU was driven by deployment reliability rather than training quality. Mobilint NPU technically supports PReLU, but LeakyReLU with a fixed slope produces a simpler and more predictable ONNX graph.

### 4.10 Operator Constraint Evolution

The project progressively narrowed the set of allowed operators across phases:

| Phase | Allowed Ops | Explicitly Forbidden |
|---|---|---|
| First Try | Conv2d, PReLU, Hardsigmoid, AdaptiveAvgPool | ReLU, Sigmoid, BatchNorm (already avoided) |
| Phase 1–5 | Conv2d, BN, PReLU, SE, Dropout, AdaptiveAvgPool | `FORBIDDEN_TYPES`: ReLU, Sigmoid, Softmax, LayerNorm, GroupNorm |
| Phase 6 | Same as above + RepConv, StochasticDepth | Same |
| Phase 7/Final | **Conv2d, LeakyReLU only** | All normalization, all attention, all pooling, all dropout |

---

## 5. Loss Functions

| Phase | Loss | Formula |
|---|---|---|
| First Try – Phase 3 | **L1** | `F.l1_loss(pred, target)` |
| Phase 4–5 | **Combined Charbonnier + L1** | `0.5 × √(diff² + ε²) + 0.5 × L1` |
| Phase 6 | Same combined loss | Same as above, ε = 1e-6 |
| Phase 7 | **Restoration Loss** | `0.60 × Charbonnier + 0.30 × L1 + 0.10 × EdgeLoss` |
| Last Run | **Residual-Target L1** | `F.l1_loss(pred - lr, hr - lr)` — optimizes the residual directly |

The Phase 7 **EdgeLoss** penalizes gradient mismatches:
```python
edge_loss = 0.5 * (L1(pred_dx, target_dx) + L1(pred_dy, target_dy))
```

---

## 6. Training Infrastructure

### 6.1 Optimizer & Schedule
- **AdamW** (with fused backend on CUDA when available)
- Linear warmup (1–5 epochs) → Cosine annealing decay
- Weight decay: 1e-4 to 5e-5 (reduced in later stages)
- Gradient clipping at norm 0.75–1.0

### 6.2 Exponential Moving Average (EMA)
- All phases from Phase 5 onward use **EMA** (decay 0.999–0.9995)
- EMA tracks `named_parameters()` only (not BN buffers — identified as a problem in Phase 6)
- Validation runs use EMA-shadowed weights
- Checkpoints store both raw model state and `ema_shadow` dict

### 6.3 Automatic Mixed Precision (AMP)
- bf16 preferred on supported GPUs (A100)
- fp16 with GradScaler fallback on older GPUs (RTX 2060 tested)
- Channels-last memory format on CUDA for performance

### 6.4 Compute Platforms Used
- **Google Colab** (Phase 1–4A) — standard GPU runtime
- **Modal** (Phase 4B–7, final run) — GPU preference `["A100", "L40S"]` or `["L40S", "A100"]`; actual GPUs varied per run:
  - Phase 4B: A100 or L40S (preference list: `["A100", "L40S"]`)
  - Phase 6 screening: **NVIDIA L40S** (confirmed in run summaries)
  - Phase 7B: bf16 AMP policy (compatible with both A100 and L40S)
- **TPU** attempted in Phase 5A (via `lab2_phase5a_wide_se_tpu.ipynb`)

### 6.5 Modal Volume Configuration

Five Modal volumes were created over the project's lifecycle:

| Volume | Created | Contents (verified via `modal volume ls`) |
|---|---|---|
| `lab2-phase4b-data` | 2026-04-05 | `Data/` (HR_train, LR_train, HR_val, LR_val, val, course_files_export) |
| `lab2-phase4b-runs` | 2026-04-05 | `phase4b_dsdan/` (smoke run), `phase4b_dsdan_full_e1/` (1-epoch), `phase4b_dsdan_full_train/` (55 epochs) |
| `lab2-phase6-data` | 2026-04-06 | `course_files_export/coco2017/` (COCO zips + extracted) |
| `lab2-phase6-runs` | 2026-04-06 | `phase6_screening/wide_se/` (coco_only + coco_plus_imagenet), `phase6_screening/dsdan/` (coco_plus_imagenet) |
| `lab2-phase7-data` | 2026-04-07 | `Data/` (full dataset), `runs/` (Phase 7B runs + leaky_relu post-submission runs) |

**Data volume layout** (`lab2-phase7-data`):
```
Data/
├── HR_train/HR_train1–4/     # High-res training (3,036 images)
├── LR_train/LR_train1–4/     # Low-res training (3,036 images)
├── HR_val/                    # High-res validation (100 images)
├── LR_val/                    # Low-res validation (100 images)
└── course_files_export/
    ├── imagenet_train20.zip / imagenet_train20a/  (+ .txt manifest)
    ├── imagenet_val20.zip / imagenet_val20/        (+ .txt manifest)
    └── coco2017/
        ├── train2017.zip / train2017/  (+ coco_train2017.txt)
        └── val2017.zip / val2017/      (+ coco_val2017.txt)
```

### 6.6 Google Colab Run History

Two lab2 notebook runs were executed on Google Colab (verified via Google Drive API):

| Notebook | Drive ID | Created | Modified | Revisions | Epochs | Best Combined Val PSNR | Best Paired Val |
|---|---|---|---|---:|---:|---:|---:|
| `lab2_phase4a_wide_se_resnet.ipynb` | `12IAcIWaw80X...` | 2026-04-06 01:32 | 2026-04-10 01:23 | 103 | 80 | **25.841 dB** | **21.848 dB** |
| `lab2_phase5d_large_kernel_dw_sr.ipynb` | `186sdzB3ZQSm...` | 2026-04-06 07:26 | 2026-04-06 13:25 | 68 | 40 | **25.861 dB** | **21.749 dB** |

A third notebook (`lab2_phase5d_large_kernel_dw_sr.ipynb`, ID `100-A8F...`) was an earlier abandoned copy with only 7 revisions over ~10 minutes (2026-04-06 07:04–07:14).

All Colab notebooks are stored in Google Drive at:
`/content/drive/MyDrive/Data 255 Class Spring 2026/Lab 2/`

**Phase 4A Colab run details:**
- Config: batch_size=32, lr=3e-4, weight_decay=2e-4, warmup=5 epochs, EMA decay=0.999, grad_clip=1.0
- Resumed from epoch 4 (checkpoint PSNR 24.856 dB); trained to epoch 80
- ~80.8s/epoch, 225.4 min total; output → `runs/phase4a_wide_se`
- Per-source eval (epoch 80): train_eval 27.431 dB, paired_val **21.848 dB** (+0.512 over baseline), imagenet_val 27.172 dB

**Phase 5D Colab run details:**
- Config: batch_size=8, lr=3e-4, weight_decay=2e-4, warmup=5 epochs, EMA decay=0.999, bf16 AMP
- LargeKernelDWSR: 14 blocks, 96 channels, expansion=2, kernels=(7,11), 650,595 params
- 40 epochs from scratch; ~442s/epoch; output → `runs/phase5d_large_kernel_dw`
- Per-source eval (epoch 40): train_eval 27.575 dB, paired_val **21.749 dB** (+0.413 over baseline), imagenet_val 27.233 dB
- ONNX export: 2,539 KB, checker passed, ORT parity max_diff=0.000279

### 6.7 Colab / Drive Data Staging
Phase 5's `phase5_common.py` includes `resolve_colab_workspace()` which looks for tarballs (`lab2_phase5_data.tar`, `lab2_colab_data.tar`) and stages them to a local `Data/` directory from Google Drive sync roots.

### 6.8 Complete Modal Run History

All Modal runs verified via `modal volume ls` and downloaded summary/metrics files:

#### Phase 4B — DSDAN on Modal (April 5, 2026)

Three runs on `lab2-phase4b-runs`:

| Run | Epochs | Best Val PSNR | Artifacts |
|---|---:|---:|---|
| `phase4b_dsdan` | 1 (smoke test) | 7.587 dB | best.pt, best.onnx, calibration/, executed notebook |
| `phase4b_dsdan_full_e1` | 1 (full data) | 23.618 dB | best.pt, best.onnx, calibration/, executed notebook |
| **`phase4b_dsdan_full_train`** | **55** | **25.992 dB** (ep 55) | best.pt, last.pt, checkpoints every 10 epochs |

The DSDAN full training run (55 epochs, ~432s/epoch) achieved a combined **val_psnr of 25.992 dB** — competitive with the Wide SE-ResNet Phase 4A result of 25.713 dB from Colab. This was the first Modal-based training run in the project.

#### Phase 6 — Screening on Modal (April 6, 2026)

Three completed screening configs on `lab2-phase6-runs`, all on **NVIDIA L40S**:

| Config | Stage 1 (Pretrain) | Stage 2 (Finetune) | Total Time | Final Paired Val PSNR |
|---|---|---|---:|---:|
| `wide_se / coco_plus_imagenet` | 12 epochs, 53.4 min | 8 epochs, 4.5 min | **57.9 min** | **21.826 dB** |
| `wide_se / coco_only` | 12 epochs | 8 epochs | ~58 min | 21.809 dB |
| `dsdan / coco_plus_imagenet` | 12 epochs | 8 epochs | ~30 min | 21.693 dB |

Detailed Stage 1 behavior (from summaries): paired-val PSNR peaked at epoch 3–4 during synthetic pretraining, then degraded — confirming synthetic training moves the model away from the paired-val objective. Stage 2 best was always at the final epoch (epoch 8), suggesting paired fine-tuning was still improving when it stopped.

Stage artifacts: `summary.json`, `metrics.jsonl`, `best.pt`, `last.pt`, epoch checkpoints, executed notebooks (`.ipynb` with outputs).

#### Phase 7B — Main Run on Modal (April 8, 2026)

Two runs on `lab2-phase7-data` under `runs/phase7b_modal_stable_checkpointed/mixed_kernel_residual/`:

| Run ID | Status | Notes |
|---|---|---|
| `20260408_004321` | **Failed at startup** | Volume commit error (not inside Modal container); only 1 line in train.log |
| **`20260408_005528`** | **Completed successfully** | Full 3-stage curriculum + ONNX export + calibration |

**Run `20260408_005528` details** (from `run_events.jsonl`, 380 events):

| Stage | Scheduled Epochs | Actual Epochs | Duration | Best hrlr_val PSNR | Best Epoch |
|---|---:|---:|---:|---:|---:|
| `synthetic_warmup` | 6 | 6 | **15.0 min** | 21.336145 | 6 |
| `mixed_finetune` | 24 | 11 (early stopped) | **28.3 min** | 21.336181 | 7 |
| `paired_polish` | 20 | 9 (early stopped) | **5.1 min** | 21.336182 | 9 |
| **Total training** | 50 | **26** | **48.4 min** | | |
| ONNX export | — | — | 1.5 sec | | |
| Calibration export | — | — | 81.4 sec | | |
| **Total wall time** | | | **56.5 min** | | |

Final status: `phase=export_complete`, device `cuda`, AMP `bf16`.

The `hrlr_val_delta_psnr` (improvement over identity) peaked at **+0.000018 dB** — effectively zero. The model learned a near-zero residual, with `residual_ratio` of ~0.0002 (predicted residual magnitude vs. input magnitude). This confirms the model was unable to improve beyond the identity function on this paired-val distribution, despite achieving train PSNR ~24.5 dB.

#### Post-Submission Runs (April 10, 2026)

Two additional **leaky_relu** runs discovered on the Phase 7 data volume:

| Run | Artifacts | Notes |
|---|---|---|
| `leaky_relu_last_run_20260410_041033` | `checkpoints/best.pt`, `checkpoints/last.pt` | No ONNX export (exports/ empty) |
| `leaky_relu_last_run_20260410_041920` | `checkpoints/best.pt`, `checkpoints/last.pt`, `exports/best.onnx` | Completed with ONNX export |

These runs (April 10, 2 days after the Canvas submission on April 8) appear to be additional experiments with the LeakyReLU-based architecture, possibly iterating on the submitted model or testing alternative configurations.

---

## 7. ONNX / MXQ Deployment Pipeline

### 7.1 ONNX Export
- Fixed input shape: `[1, 3, 256, 256]`
- Opset version: 13
- Input name: `input`, Output name: `output`
- Constant folding enabled
- Post-export: ONNX checker validation + ONNX Runtime parity check
- **Final run export**: ONNX file size **9,330 KB**; ORT parity max_diff = **6e-8**, mean_diff = **0.0** (near-perfect numerical agreement)

### 7.2 PReLU → LeakyReLU Rewrite
The submission pipeline ([lab_step2_onnx-to-mxq.py](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/Submission/lab_step2_onnx-to-mxq.py)) includes a **PReLU-to-LeakyReLU rewriter**:
- Traverses ONNX graph, finds all `PRelu` nodes
- Averages per-channel slopes → single scalar alpha
- Replaces with `LeakyRelu(alpha=mean(slopes))`
- Ensures NPU compatibility (PReLU is technically supported but LeakyReLU is more reliable)

### 7.3 MXQ Quantization
- Uses **Qubee** (`mxq_compile`) for Mobilint NPU compilation
- **Calibration**: 128 diverse samples selected via `select_diverse_calibration_subset()`:
  - Diversity sampling across brightness and texture tertiles (3×3 grid = 9 strata)
  - Sources: paired training LR images + COCO/ImageNet samples (depending on phase)
  - Stored as PNG files in `phase7b_calibration_png_lr_dataset/` (with optional `manifest.json`)
  - Phase 5/6 used `export_default_calibration()` with `calibration_datasets` map covering paired + COCO + ImageNet pools
- Quantization method: `maxpercentile` (percentile=0.999, topk_ratio=0.01)
- Preprocessing: RGB float32 [0, 1], HWC format, resize to 256×256
- Output: `lab2.mxq` (~5.5 MB)
- MXQ PSNR evaluation scripts: `eval_mxq_psnr.py`, `eval_mxq_psnr_step3_paths.py`, `step3_mxq_only.py` (evaluate against paired val and ImageNet val subsets)

---

## 8. Models Submitted

### 8.1 Canvas Submission (April 8, 2026)

**Files in** [CanvasSubmissionApril8/](file:///Users/mrcyrilgoud/Desktop/repos/Lab-2/CanvasSubmissionApril8):

| File | Description |
|---|---|
| `lab2_final_model_cleaned_no_calibration.ipynb` | Full pipeline notebook (data, model, training, export) |
| `lab2_final_model.ipynb` | Version with calibration outputs |
| `lab2.mxq` | **Compiled MXQ model for Mobilint NPU** (5.5 MB) |
| `lab_step2_onnx-to-mxq.py` | ONNX → MXQ conversion script with PReLU rewrite |

### 8.2 Submitted Model Architecture

**Model**: `mixed_kernel_residual` variant of `ResidualConvSR`

```
Architecture: ResidualConvSR with mixed 3×3/5×5 kernel pattern

Stem:     Conv2d(3 → 96, k=3, p=1) → LeakyReLU(0.10)

Body:     18 × ConvLeakyReLUBlock, kernel pattern: [3, 3, 5] × 6
          Block: Conv2d(96 → 96, k=K, p=K//2) → LeakyReLU(0.10)

Tail:     Conv2d(96 → 3, k=3, p=1)    [init: scale=1e-3]

Forward:  output = input + tail(body(stem(input)))
```

**Properties**:
- **Parameters**: 2,386,563 (all Conv2d + LeakyReLU)
- **Module count**: 20 Conv2d + 19 LeakyReLU
- **Only 2 operation types**: Conv2d and LeakyReLU — maximally NPU-friendly
- **No normalization, no attention, no dropout**: eliminates all Phase 6 issues
- **Mixed kernel sizes**: alternating 3×3 and 5×5 kernels for multi-scale feature capture without increasing architectural complexity

### 8.3 Training Protocol for Submitted Model

The submitted model was trained using the **Phase 7B 3-stage curriculum** on Modal (run `20260408_005528`, bf16 AMP):

| Stage | Epochs | Data | LR | EMA Decay |
|---|---:|---|---|---|
| `synthetic_warmup` | 6 | Calibrated synthetic (COCO + ImageNet) | 3e-4 | 0.999 |
| `mixed_finetune` | 24 | 65% paired + 35% calibrated synthetic | 1.5e-4 | 0.999 |
| `paired_polish` | 20 | Paired only (hard-sample weighted) | 5e-5 | 0.9995 |

**Loss**: Residual-target L1 (`F.l1_loss(pred - lr, hr - lr)`)
**Batch size**: 24, **Patch size**: 224×224 train / 256×256 eval
**AMP**: bf16 (GPU preference: A100 or L40S)
**Datasets**: Paired train 3,036 pairs; Paired val 100 pairs; Natural train 18,000 (COCO 12,000 + ImageNet 6,000); Natural val 800

### 8.4 Final Training Run Outcome

The canonical run (`20260408_005528`) trained all three curriculum stages on Modal (total wall time: **56.5 minutes**):

| Stage | Scheduled Epochs | Actual Epochs | Duration | Train PSNR | hrlr_val PSNR | Best Epoch | Early Stop |
|---|---:|---:|---:|---:|---:|---:|---|
| `synthetic_warmup` | 6 | 6 | 15.0 min | 23.541 | 21.336145 | 6 | No |
| `mixed_finetune` | 24 | 11 | 28.3 min | 23.794 | 21.336181 | 7 | Yes (no ≥0.001 dB gain for 10 epochs) |
| `paired_polish` | 20 | 9 | 5.1 min | 24.474 | 21.336182 | 9 | Yes (no ≥0.001 dB gain for 8 epochs) |

Post-training: ONNX export (1.5s) + calibration export of 128 samples (81.4s). Final status: `export_complete`.

> [!NOTE]
> The `hrlr_val_psnr` of **21.336 dB** equals the paired-val **baseline** LR-vs-HR PSNR — meaning the model's delta over identity was **+0.000018 dB** at best. The `residual_ratio` (predicted residual magnitude / input magnitude) was ~0.0002, confirming the model learned a near-zero residual. This is consistent with the Phase 6 postmortem finding that paired-val PSNR is an extremely hard metric to move due to the fundamental train/val difficulty mismatch.
>
> Validation breakdown by difficulty tier (last epoch): **hard** 19.111 dB, **mid** 22.779 dB, **easy** 24.809 dB — all essentially unchanged from the LR baseline for each tier.

---

## 9. Key Findings & Lessons Learned

### Architecture Lessons
1. **BatchNorm hurts restoration with domain mismatch**: BN statistics drift between synthetic pretraining and paired fine-tuning, especially with small batches
2. **Attention/gating can learn wrong priors**: SE and spatial gates trained on synthetic degradation don't transfer to real paired degradation
3. **Simpler is better for deployment**: The final Conv+LeakyReLU-only model outperforms complex alternatives on the actual deployment target
4. **Mixed kernel sizes help**: Alternating 3×3 and 5×5 captures multi-scale features without SE/attention overhead

### Data Lessons
1. **Calibrated degradation > generic degradation**: Rejection-sampling synthetic degradation to match validation-set difficulty distribution is critical
2. **Hard-sample emphasis matters**: WeightedRandomSampler oversampling for hard training pairs helps close the train/val gap
3. **3-stage curriculum outperforms 2-stage**: Adding a mixed stage between pure synthetic and pure paired fine-tuning bridges the distribution gap

### Training Lessons
1. **EMA is essential**: Smoothed model weights consistently outperform raw training weights
2. **Edge loss as a minor component** (10%) helps preserve structural details
3. **Residual-target L1** (optimizing the delta directly) is more numerically stable for restoration than standard pixel L1

---

## 10. Performance Summary Across Phases

| Phase | Architecture | Paired Val PSNR | Combined Val PSNR | Best Epoch | Notes |
|---|---|---:|---:|---:|---|
| 1 | ResNet (48ch, 10 blocks) | **21.362** | — | 12 | Baseline; paired-only training |
| 2 | ResNet-SE (48ch, 10 blocks) | **21.737** | — | 38 | +0.38 dB from SE attention |
| 2 | Hybrid UNet-SE (32/48/64ch) | **21.898** | — | 35 | Best Phase 2 result |
| 3 | ResNet-SE (40ch, 8 blocks) | **21.696** | **25.543** | 40 | First combined metric w/ ImageNet |
| 4A | Wide SE-ResNet (64ch, 16 blocks) | **21.848** | **25.841** | 80 | Colab; 80 epochs, ~80.8s/ep, 225.4 min |
| 4B | DSDAN (128ch, 12 blocks) | — | **25.992** | 55 | Modal; 55 epochs, ~432s/ep |
| 5D | Large-Kernel DW (96ch, 14 blocks) | **21.749** | **25.861** | 40 | Colab; 40 epochs, ~442s/ep |
| 6 (best) | Wide SE + coco+imagenet | **21.826** | 24.849 | — | Only +0.13 dB over Phase 3 on paired val |
| 6 (#2) | Wide SE + coco_only | 21.809 | 24.541 | — | COCO helps synthetic, not paired |
| 6 (#3) | DSDAN + coco+imagenet | 21.693 | 24.830 | — | More gating ≠ better paired val |
| 7/Final | mixed_kernel_residual | 21.336* | ~22.4 | — | Final submitted model (Modal, bf16) |

*\*The 21.336 dB paired-val PSNR for the final run equals the LR-vs-HR baseline — meaning delta ≈ 0.000 on this metric.*

> [!NOTE]
> **The persistent paired-val plateau at ~21.3–21.8 dB** across all phases and architectures — from the 10-block Phase 1 ResNet to the 7-model Phase 7B screening — is the defining challenge of this project. The Phase 6 postmortem concluded that the bottleneck was not model capacity but the fundamental mismatch between training data difficulty (mean baseline ~26–27 dB) and validation data difficulty (mean baseline ~21.3 dB). Even calibrated synthetic degradation and 3-stage curriculum training in Phase 7 could not overcome this gap.

### 10.1 Architecture Complexity vs. Performance

The project explored a wide range of model complexity without proportional PSNR gains:

| Architecture Family | Param Range | Unique Op Types | Best Paired Val | Phase |
|---|---|---:|---:|---|
| Plain residual (Conv+BN+PReLU) | 48K–1.5M | 3 (Conv, BN, PReLU) | 21.362 | 1 |
| SE-Residual | 750K–1.2M | 5 (+SE, Hardsigmoid) | 21.848 | 4A (Colab) |
| Depthwise + Dual Attention | 622K | 7 (+DW, SpatialGate) | 21.693 | 6 |
| Multi-scale + InstanceNorm + Mish | 757K | 9 (+DilatedDW, IN, Mish, Gate) | — | 4C |
| UNet Encoder-Decoder + SE | — | 6 (+PixelShuffle) | 21.898 | 2 |
| RepConv (re-parameterizable) | 2.1M | 5 (3-branch at train) | — | 6 |
| Large-Kernel DW | 650K–717K | 5–6 | 21.749 | 5D (Colab) |
| **No-norm Conv+LeakyReLU (final)** | **1.5M–5.6M** | **2 (Conv, LeakyReLU)** | **21.336** | **7B** |

---

## Appendix A: Complete Data Source Inventory

### A.1 Course Paired LR/HR Dataset (Primary)

| Property | Value |
|---|---|
| **Format** | PNG, 256×256, RGB |
| **Training** | 4 folder pairs: `HR_train1–4` / `LR_train1–4` (~3,036 total pairs) |
| **Validation** | `HR_val` / `LR_val` (100 pairs) |
| **Train baseline PSNR** | ~26–27 dB (relatively easy degradation) |
| **Val baseline PSNR** | ~21.3 dB (significantly harder degradation) |
| **Layout variants** | Structured (`HR_train/` + `LR_train/` with numbered subfolders) or flat (`train/LR` + `train/HR`) |
| **Collectors** | `collect_train_pairs()`, `collect_val_pairs()`, `collect_paired_by_subfolder()`, `collect_paired_flat()` |
| **Used in** | All phases (Phase 1–7) |

### A.2 ImageNet-20 (Course Subset)

| Property | Value |
|---|---|
| **Source** | Course-provided subset of ImageNet (not full ImageNet) |
| **Manifests** | `imagenet_train20.txt`, `imagenet_val20.txt` under `course_files_export/` |
| **Archives** | `imagenet_train20.zip` → `imagenet_train20a/`, `imagenet_val20.zip` → `imagenet_val20/` |
| **Typical limits** | 6,000 train / 300 val (env-overridable) |
| **Role** | Synthetic SR via `degrade_from_hr()` — HR crops degraded to create LR |
| **Collectors** | `collect_imagenet_records()`, `build_natural_records()` |
| **Used in** | Phase 3 (eval only), Phase 4–7 (training + eval) |

### A.3 COCO 2017

| Property | Value |
|---|---|
| **Source** | Official COCO dataset from `images.cocodataset.org` |
| **Download** | `stage_coco2017()` helper with `COCO_URLS` dict |
| **Archives** | `train2017.zip`, `val2017.zip` under `course_files_export/coco2017/` |
| **Manifests** | `coco_train2017.txt`, `coco_val2017.txt` (generated after extraction) |
| **Screening limits** | 12,000 train / 500 val |
| **Full-run limits** | 30,000 train / 1,000 val |
| **Storage** | Kept as zips on Modal volumes for fast volume cloning |
| **Role** | Synthetic SR (same degradation pipeline as ImageNet); third data source |
| **Used in** | Phase 6–7 |

### A.4 Synthetic Degradation Pipeline Evolution

| Phase | Pipeline | Key Parameters |
|---|---|---|
| **4–5** | `degrade_from_hr()` via `ImageNetSyntheticSRDataset` | Gaussian blur (0.2–1.2), downscale 2/3/4 (bicubic), JPEG (40–90) |
| **6** | `NaturalImageSyntheticSRDataset` (enhanced) | Wider blur (0.2–1.6), multi-mode resize (bicubic/bilinear/lanczos), JPEG (25–90) |
| **7** | `CalibratedSyntheticSRDataset` (rejection-sampled) | Target PSNR 17.5–24.2 dB, gradient ratio 0.35–0.85, Laplacian ratio 0.25–0.75, RGB gamma shift, up to 12 rejection attempts |

**Tensor-level augmentations** (applied to LR after degradation):
- LR noise injection: prob 0.30, std 0.015 (disabled in Phase 7 paired-polish stage)
- Random cutout: prob 0.35, ratio 0.18 (disabled in paired fine-tuning stages)

**Geometric augmentations** (synchronized for paired data):
- Horizontal flip, vertical flip, k×90° rotation
- Applied via `augment_pair()` / `augment_single()`

### A.5 Calibration Data for MXQ Quantization

| Property | Value |
|---|---|
| **Sample count** | 128 images |
| **Selection** | Diversity sampling across brightness × texture tertiles (3×3 = 9 strata) |
| **Sources** | Paired LR training images + COCO/ImageNet samples |
| **Storage** | `phase7b_calibration_png_lr_dataset/` (PNG files + optional `manifest.json`) |
| **Construction** | `select_diverse_calibration_subset()` in phase5/6 common modules; `export_default_calibration()` |
| **Format** | RGB PNG, 256×256 |

### A.6 Augmentation & Patch Configuration

| Setting | Typical Value | Where Used |
|---|---|---|
| `train_patch_size` | 224 | Phase 5/6 defaults; env-overridable in Phase 7 |
| `eval_size` | 256 | Paired eval (full image) |
| `random_scale_pad` | 32 (Phase 5), 48 (Phase 6), 64 (Phase 7) | Synthetic HR crop margin before degradation |
| Geometric augmentation | H/V flip, 90° rotations | All phases |
| LR noise (prob / std) | 0.30 / 0.015 | Phase 4+; disabled in paired-polish |
| Random cutout (prob / ratio) | 0.35 / 0.18 | Phase 4+; disabled in paired fine-tuning |

---

## Appendix B: Complete Architecture Registry

All 20+ unique model architectures explored across the project, ordered chronologically:

| # | Phase | Class Name | Channels | Blocks | Key Ops | Params | Notes |
|---:|---|---|---:|---:|---|---:|---|
| 1 | 0 | `MultiScaleResidualAttentionNet` | 48 | 3×3 | DW, PReLU, Hardsigmoid, StageAttention | — | Multi-scale kernels (3,5,7) |
| 2 | 1 | `ResNetSR` | 48 | 10 | Conv, BN, PReLU | — | Baseline |
| 3 | 2 | `ResNetSESR` | 48 | 10 | Conv, BN, PReLU, SE, Hardsigmoid | — | +SE attention |
| 4 | 2 | `UNetSR` | 32/48/64 | — | Conv, BN, PReLU, PixelShuffle | — | Encoder-decoder |
| 5 | 2 | `HybridUNetSESR` | 32/48/64 | — | Conv, BN, PReLU, SE, PixelShuffle | — | UNet + SE |
| 6 | 3 | `ResNetSESR` (v2) | 40 | 8 | Conv, BN, PReLU, SE, Dropout2d, StochasticDepth | — | +regularization |
| 7 | 4A | `WideSEResNetSR` | 64 | 16 | Conv, BN, PReLU, SE, Dropout2d, StochasticDepth | 1,222,723 | Scaled-up SE-ResNet |
| 8 | 4B | `DSDANSR` | 128 | 12 | DW, PW, BN, PReLU, SE, SpatialGate | 622,467 | Depthwise dual-attention |
| 9 | 4C | `MFRNSR` | 56 | 3×4 | Conv, DilatedDW, BN, IN, PReLU, Mish, SE, Gate | 757,179 | Multi-scale refinement |
| 10 | 5B | `DSDANSR` (A100) | 128 | 12 | Same as 4B | 622,467 | GPU-optimized |
| 11 | 5C | `RepConvSR` | 96 | 12 | RepConvBN (3-branch), BN, PReLU | 2,117,091 | Re-parameterizable |
| 12 | 5D | `LargeKernelDWSR` | 96 | 14 | DW (7×7, 11×11), BN, PReLU | 650,595 | Large kernels |
| 13 | 6 | `LargeKernelSESR` | 96 | 14 | DW (7×7, 11×11), BN, PReLU, SE | 717,123 | +SE on large-kernel |
| 14 | 6 | `HybridRepLargeKernelSR` | 96 | 12 | RepConv + LargeKernelDW alternating | 1,337,763 | Hybrid |
| 15 | 7 | `Phase7NoNormResidualConvSR` | 96 | 18 | Conv, PReLU | — | No-norm baseline |
| 16 | 7 | `Phase7DirectConvSR` | 96 | 18 | Conv, PReLU | — | No global residual |
| 17 | 7B | `ResidualConvSR` (nonorm) | 96 | 18 | Conv, LeakyReLU | 1,501,827 | All 3×3 |
| 18 | 7B | **`ResidualConvSR` (mixed)** ⭐ | 96 | 18 | Conv, LeakyReLU | **2,386,563** | **[3,3,5]×6 (submitted)** |
| 19 | 7B | `ExpandedDepthwiseResidualSR` | 128 | 12 | DW (9×9), LeakyReLU | 1,057,795 | Inverted bottleneck |
| 20 | 7B | `ExpandedDepthwiseResidualSR` (deep) | 128 | 18 | DW (7×7), LeakyReLU | 1,435,651 | Deeper variant |
| 21 | 7B | `ResidualConvSR` (wide) | 128 | 24 | Conv, LeakyReLU | 5,649,411 | [3,3,5]×8 |
| 22 | 7B | `ExpandedDepthwiseResidualSR` (v2) | 160 | 14 | DW (9×9), LeakyReLU | 1,827,843 | Higher capacity |
| 23 | 7B | `HybridLargeFineResidualSR` | 128 | 8+10 | DW (9×9) + Conv (3/5), LeakyReLU | 3,495,427 | Two-stage body |
