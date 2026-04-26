# Lab 2 — What Worked, What Didn't, and Next Steps

> [!NOTE]
> This Lab 3 repository keeps selected Lab 2 reference material locally. Preserved Lab 2 run artifacts now live under `historical/lab2/last_run/`; some older path names mentioned in this writeup refer to the original Lab 2 workspace layout.

> Post-mortem analysis synthesized from 8 development phases, 20+ architectures,
> 5 compute platforms, and the April 8 Canvas submission.

---

## Executive Summary

Across ~2 weeks of iterative development, the project explored progressively
more sophisticated model architectures, data pipelines, and training protocols
for same-resolution (256x256) image restoration. The **best paired-val PSNR
ever achieved was ~21.9 dB** (Phase 2 Hybrid UNet-SE), only **+0.56 dB above
the Phase 1 baseline** (21.36 dB). The final submitted model
(`mixed_kernel_residual`) produced **+0.000 dB delta over identity** on paired
validation — a near-zero residual.

The defining insight: the bottleneck was never model capacity or architecture
sophistication. It was the **~5-6 dB difficulty gap between training data
(baseline ~26-27 dB) and validation data (baseline ~21.3 dB)**, which no
combination of architecture, loss function, or curriculum could bridge.

---

## Part 1: What Worked Best

### 1.1 Removing BatchNorm (Phase 7)

**Impact: High | Confidence: High**

The Phase 6 postmortem identified BatchNorm as a persistent source of domain
shift between synthetic pretraining and paired fine-tuning. Phase 7's radical
decision to strip all normalization layers was the single most important
architectural insight of the project:

- Eliminated BN statistics drift at small batch sizes (screening used batch 4)
- Removed the EMA-not-tracking-BN-buffers bug that plagued Phase 5-6
- Simplified ONNX export and NPU deployment (no running mean/var state)
- Resulted in the cleanest deployment artifact: only Conv2d + LeakyReLU ops

Even though the final model didn't beat earlier phases on paired-val PSNR, the
no-norm architecture was the **correct design decision for the deployment
target** (Mobilint NPU via MXQ).

### 1.2 Calibrated Synthetic Degradation (Phase 7)

**Impact: Medium-High | Confidence: Medium-High**

The rejection-sampling approach that matched synthetic degradation to
validation-set difficulty distribution was a sound methodological advance:

- Degradation profiling quantified the train/val gap (26.76 vs 21.34 dB
  baseline PSNR)
- Target PSNR range (17.5-24.2 dB) centered on the actual validation
  distribution
- Gradient and Laplacian ratio targets ensured structural similarity, not just
  PSNR-level matching

This was the right idea even though the final metrics didn't reflect it —
the synthetic degradation still couldn't replicate the *specific* artifact
types present in the paired validation set.

### 1.3 SE Channel Attention (Phase 2-4)

**Impact: Medium | Confidence: High**

Squeeze-and-Excitation attention produced the most consistent gains across
phases:

| Model | Paired Val PSNR | Delta vs Baseline |
|---|---:|---:|
| Phase 1 ResNet (no SE) | 21.362 | — |
| Phase 2 ResNet-SE | 21.737 | +0.375 |
| Phase 2 Hybrid UNet-SE | 21.898 | +0.536 |
| Phase 4A Wide SE-ResNet (Colab, 80ep) | 21.848 | +0.486 |
| Phase 6 Wide SE + COCO+ImageNet | 21.826 | +0.464 |

SE attention was the one architectural addition that reliably improved
paired-val PSNR by +0.4-0.5 dB. The gains were modest but consistent.

### 1.4 Global Residual Learning (All Phases)

**Impact: High | Confidence: High**

The `output = input + model(input)` formulation was present from Phase 0 and
never abandoned. It was unequivocally the right choice for same-resolution
restoration:

- The model only needs to predict the *delta*, not reconstruct the full image
- Numerically stable (identity initialization via small tail weights)
- Phase 7's residual-target L1 loss (`F.l1_loss(pred - lr, hr - lr)`)
  directly optimized this delta

### 1.5 EMA Model Averaging (Phase 5+)

**Impact: Medium | Confidence: High**

Exponential Moving Average (decay 0.999-0.9995) consistently outperformed raw
training weights. Every phase from 5 onward used EMA for validation and export.

### 1.6 Mixed Kernel Sizes (Phase 7B)

**Impact: Low-Medium | Confidence: Medium**

The `[3, 3, 5] x 6` kernel pattern in the submitted model captured multi-scale
features without any attention or pooling overhead. At 2.39M params it was a
reasonable trade-off between capacity and simplicity.

### 1.7 ONNX/MXQ Deployment Pipeline

**Impact: High (for deliverable) | Confidence: High**

The end-to-end deployment pipeline was polished and reliable:

- ONNX export with opset 13, max_diff = 6e-8 (near-perfect parity)
- PReLU-to-LeakyReLU graph rewriter for NPU compatibility
- Diversity-sampled 128-image calibration set (brightness x texture tertiles)
- MXQ compilation via Qubee with maxpercentile quantization
- Final `lab2.mxq` artifact at ~5.5 MB

---

## Part 2: What Worked Worst

### 2.1 Synthetic Pretraining for Paired-Val Improvement

**Impact: Negative | Confidence: High**

This was the project's most persistent false lead. Across Phase 4-7, synthetic
data (ImageNet, COCO, calibrated degradation) consistently *hurt* paired-val
PSNR during pretraining:

| Evidence | Finding |
|---|---|
| Phase 6 screening | Paired-val peaked at epoch 3-4 of Stage 1, then degraded |
| Phase 7B run | `hrlr_val_psnr` flat at 21.336 across all 26 epochs |
| Combined vs paired | Combined-val improved (~25.8 dB) while paired-val stagnated (~21.3-21.8) |

The synthetic degradation pipeline (Gaussian blur + downsample + JPEG) did not
produce artifacts that matched the paired validation set's degradation profile.
The model learned to undo *generic* degradation while being unable to fix
*specific* paired-val artifacts.

### 2.2 Architecture Over-Engineering (Phase 4C, 5C, 6 Hybrid)

**Impact: Wasted Effort | Confidence: High**

Several architecturally complex models consumed significant development time
with no measurable benefit:

| Model | Unique Op Types | Complexity | Paired Val Result |
|---|---:|---|---|
| MFR-Net (Phase 4C) | 9 | InstanceNorm, Mish, dilated DW, gating | Never evaluated |
| RepConvSR (Phase 5C) | 5 | 3-branch re-parameterizable | Never evaluated |
| HybridRepLargeKernelSR (Phase 6) | Mixed | RepConv + LargeKernelDW | Not screened |

MFR-Net in particular was the most architecturally complex model explored
(InstanceNorm, Mish, dilated depthwise convolutions, per-stage gating) — and
it was never even fully evaluated against the baseline.

### 2.3 BatchNorm with Small Batches (Phase 1-6)

**Impact: Negative | Confidence: High**

BatchNorm was present in every model from Phase 1-6 (33-61 BN layers per
model). With Phase 6 screening running at batch size 4, BN statistics were
noisy and unreliable. Combined with EMA not tracking BN running buffers, this
created a systematic evaluation error: the EMA-averaged model weights were
paired with stale BN statistics.

This bug was not identified until the Phase 6 postmortem and likely suppressed
results in earlier phases as well.

### 2.4 The 3-Stage Curriculum (Phase 7B Final Run)

**Impact: Ineffective | Confidence: High**

Despite being the most sophisticated training protocol in the project, the
3-stage curriculum produced the worst paired-val result:

| Stage | Scheduled | Actual | Best hrlr_val PSNR | Delta vs Baseline |
|---|---:|---:|---:|---:|
| synthetic_warmup | 6 ep | 6 ep | 21.336145 | +0.000 |
| mixed_finetune | 24 ep | 11 ep (early stop) | 21.336181 | +0.000 |
| paired_polish | 20 ep | 9 ep (early stop) | 21.336182 | +0.000 |

The model learned a near-zero residual (residual_ratio ~0.0002). Early stopping
triggered in both later stages after failing to gain even 0.001 dB. The
ambitious curriculum design was undermined by the fundamental data distribution
mismatch.

### 2.5 Excessive Phase Proliferation

**Impact: Wasted Effort | Confidence: Medium-High**

The project explored 8 phases, 20+ architectures, and ~15 notebooks. Many of
these were incremental variations that didn't test meaningfully different
hypotheses:

- Phase 4 had 3 sub-variants (4A, 4B, 4C) exploring width/attention/multi-scale
- Phase 5 had 4 sub-variants across 3 compute platforms
- Phase 7B had 7 model variants in the registry, most never fully trained

A more focused approach — identifying the train/val distribution mismatch early
and spending cycles on data analysis rather than architecture search — would
have been more productive.

### 2.6 Dropout and Stochastic Depth in Restoration (Phase 3-6)

**Impact: Slightly Negative | Confidence: Medium**

The Phase 6 postmortem flagged that Dropout2d and StochasticDepth hurt precise
residual correction during paired fine-tuning with limited data. For a
restoration task where the model needs to learn fine-grained pixel corrections,
randomly dropping features or entire blocks during training was
counterproductive.

---

## Part 3: Performance Ranking (All Phases)

### Paired-Val PSNR Leaderboard

| Rank | Phase | Architecture | Paired Val PSNR | Training | Compute |
|---:|---|---|---:|---|---|
| 1 | 2 | Hybrid UNet-SE | **21.898** | 35 ep, paired only | — |
| 2 | 4A | Wide SE-ResNet (64ch, 16 blk) | **21.848** | 80 ep, Colab | Colab GPU |
| 3 | 6 | Wide SE + COCO+ImageNet | **21.826** | 20 ep (2-stage) | Modal L40S |
| 4 | 6 | Wide SE + COCO only | **21.809** | 20 ep (2-stage) | Modal L40S |
| 5 | 5D | Large-Kernel DW (96ch, 14 blk) | **21.749** | 40 ep, Colab | Colab GPU |
| 6 | 2 | ResNet-SE (48ch, 10 blk) | **21.737** | 38 ep, paired only | — |
| 7 | 3 | ResNet-SE v2 (40ch, 8 blk) | **21.696** | 40 ep | — |
| 8 | 6 | DSDAN + COCO+ImageNet | **21.693** | 20 ep (2-stage) | Modal L40S |
| 9 | 1 | ResNet (48ch, 10 blk) | **21.362** | 12 ep, paired only | — |
| **10** | **7B** | **mixed_kernel_residual (submitted)** | **21.336** | **26 ep (3-stage)** | **Modal A100/L40S** |

The submitted model ranked **last** on paired-val PSNR. The best-performing
model (Phase 2 Hybrid UNet-SE) was the 5th architecture explored and used the
simplest training protocol (paired data only, no synthetic augmentation).

### Key Takeaway

**More data, more stages, more sophistication ≠ better paired-val PSNR.**
The top 3 results all used simpler training protocols with fewer stages. The
correlation between training complexity and paired-val performance was
*negative*.

---

## Part 4: Root Cause Analysis

### Why Did the Final Model Fail to Improve Over Identity?

**Primary cause: Removing normalization + SE attention eliminated the
mechanisms that earlier models used to achieve their modest gains.**

The Phase 7 design rationale was sound for NPU deployment but sacrificed the
components that actually helped:

1. **SE attention** provided +0.4-0.5 dB consistently (Phase 2-6)
2. **BatchNorm** (despite its issues) provided training stability that the
   no-norm models lacked
3. The combination of *removing all helpful components* while *adding an
   untested training curriculum* meant the model had no proven pathway to
   improvement

### Why Did the Train/Val Gap Persist?

1. **Different degradation types**: Training pairs had relatively mild
   degradation (~26-27 dB baseline) while validation pairs had severe
   degradation (~21.3 dB baseline)
2. **100-sample validation set**: Too small to reliably measure sub-dB
   improvements; high variance in the metric
3. **No analysis of degradation *type***: The project profiled PSNR/gradient/
   Laplacian statistics but never analyzed *what kind* of degradation was
   present (noise type, blur kernel, compression artifacts, etc.)
4. **Overfitting to training distribution**: Models consistently achieved
   24-27 dB on training eval but couldn't generalize to the harder validation
   distribution

---

## Part 5: Best Possible Next Steps

### 5.1 Immediate (Could Improve Submitted Results)

#### A. Retrain Wide SE-ResNet Without BN, With LeakyReLU

Combine the best of both worlds:

- **Architecture**: Wide SE-ResNet (64ch, 16 blocks) — the best paired-val
  performer at scale
- **Modifications**: Replace BatchNorm with weight standardization or no
  normalization; keep SE attention; use LeakyReLU instead of PReLU
- **Training**: Paired data only, 80+ epochs (replicating the Phase 4A Colab
  protocol that achieved 21.848 dB)
- **Rationale**: SE attention was the one component that consistently helped;
  the Phase 7 decision to remove it along with BN was overly aggressive

```
Expected impact: ~21.8-22.0 dB paired-val PSNR
NPU compatibility: SE uses AdaptiveAvgPool + Hardsigmoid — verify MXQ support
```

#### B. Train Directly on Paired Data (Skip Synthetic Stages)

The leaderboard shows that synthetic pretraining never helped paired-val PSNR:

- Use paired data only from the start
- Longer training (80-100 epochs) with cosine decay
- Hard-sample weighting from epoch 1
- This is essentially the Phase 4A protocol, which remains the best
  large-scale result

#### C. Increase Paired Fine-Tuning Duration

Phase 6 screening showed Stage 2 (paired fine-tuning) was still improving at
its final epoch. The Phase 7B paired_polish stage early-stopped after 9 epochs
with a 0.001 dB threshold — this threshold may be too aggressive:

- Lower early-stopping threshold to 0.0001 dB or remove it entirely
- Run paired_polish for the full 20 epochs
- Use a lower learning rate (1e-5) to allow finer adjustments

### 5.2 Medium-Term (Architectural Improvements)

#### D. Paired-Val Degradation Analysis

Before any more architecture work, systematically analyze *what* makes the
validation set harder:

- Compute per-image degradation fingerprints (noise level, blur kernel
  estimation, compression artifact detection)
- Compare training vs validation degradation *types*, not just difficulty
  levels
- Identify the top-10 hardest validation images and visually inspect
  LR vs HR differences
- Use this analysis to construct a targeted synthetic degradation pipeline

#### E. Test-Time Augmentation (TTA)

A zero-cost improvement at inference:

- Apply 8 geometric augmentations (4 rotations x 2 flips) at test time
- Average the (inverse-augmented) outputs
- Typically yields +0.1-0.3 dB for free on restoration models
- Fully compatible with the existing ONNX pipeline (run inference 8 times,
  average)

#### F. Self-Ensemble / Model Soup

Average weights from the best checkpoints across phases:

- Phase 4A Wide SE-ResNet (21.848 dB)
- Phase 2 Hybrid UNet-SE (21.898 dB)
- Phase 5D Large-Kernel DW (21.749 dB)

If architectures match (or can be made to match), weight averaging often
outperforms any single checkpoint.

### 5.3 Longer-Term (Fundamental Improvements)

#### G. Contrastive / Perceptual Loss Components

The project used only pixel-space losses (L1, Charbonnier, Edge). Adding a
lightweight perceptual component could help bridge the distribution gap:

- Feature-matching loss using early VGG layers (low computational cost)
- Gradient-domain loss (already explored as EdgeLoss at 10% weight; try higher)
- Frequency-domain loss (FFT-based, penalizing high-frequency differences)

#### H. Larger/More Representative Validation Proxy

The 100-image validation set is a poor signal for optimization:

- Create a "hard subset" from training data that matches validation difficulty
  (20.0-22.5 dB baseline PSNR range) — the degradation profiling already
  identified these images
- Use this as a development validation set for architecture search
- Reserve the official 100-image val set for final evaluation only
- This addresses the core issue: you can't optimize for a distribution you
  only see 100 samples from

#### I. Curriculum Learning on Difficulty, Not Data Source

Instead of staging by data source (synthetic → mixed → paired), stage by
difficulty:

- Stage 1: Easy training pairs (baseline > 28 dB) + easy synthetic
- Stage 2: Medium pairs (24-28 dB) + medium synthetic
- Stage 3: Hard pairs (< 24 dB) + hard synthetic matching val distribution
- This gradually shifts the model toward the validation difficulty without
  the domain-shift problem of switching from synthetic to paired data

---

## Part 6: Lessons for Future Projects

| Lesson | Evidence |
|---|---|
| **Analyze data before architecture search** | The 5-6 dB train/val gap was measurable from Day 1 but wasn't diagnosed until Phase 6 |
| **Simpler training protocols win** | Top paired-val results used paired-only training with no curriculum |
| **Don't remove components that help** | SE attention consistently gave +0.4-0.5 dB; removing it in Phase 7 was premature |
| **Deployment constraints ≠ training constraints** | BN can be used during training and fused/removed at export time |
| **100 samples is not enough for reliable optimization** | Sub-dB improvements on 100 images are within noise; need proxy metrics |
| **Compute spent on architecture search had diminishing returns** | 20+ architectures, <0.6 dB total spread on paired-val |
| **Negative results are still results** | The synthetic pretraining finding (it hurts paired-val) is a valuable insight for the field |

---

## Appendix: Submission Artifact Summary

| Artifact | Location | Status |
|---|---|---|
| Final notebook | `CanvasSubmissionApril8/lab2_final_model_cleaned_no_calibration.ipynb` | Submitted April 8 |
| MXQ model | `CanvasSubmissionApril8/lab2.mxq` (~5.5 MB) | Submitted April 8 |
| ONNX converter | `CanvasSubmissionApril8/lab_step2_onnx-to-mxq.py` | Submitted April 8 |
| Training run | `historical/lab2/last_run/phase7b_.../20260408_005528/` | 56.5 min on Modal |
| Best checkpoint | `historical/lab2/last_run/best.pt` | mixed_kernel_residual, 2.39M params |
| ONNX export | `historical/lab2/last_run/.../exports/best.onnx` (~9.3 MB) | ORT parity 6e-8 |
| Post-submission runs | `leaky_relu_last_run_20260410_*` (on Modal volume) | April 10, 2 days after submission |
