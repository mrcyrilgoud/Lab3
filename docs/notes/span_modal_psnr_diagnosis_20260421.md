# SPAN Modal PSNR Diagnosis

Repro command:

```bash
python3 /Users/mrcyrilgoud/Desktop/repos/Lab3/src/scripts/diagnose_span_modal_gap.py
```

## Scope

This compares the current SPAN notebook path in [experiments/SPAN NPU v1/lab3_span_npu_v1_modal.ipynb](/Users/mrcyrilgoud/Desktop/repos/Lab3/experiments/SPAN%20NPU%20v1/lab3_span_npu_v1_modal.ipynb) against the baseline Modal path in [src/pipelines/lab3_pipeline_lib.py](/Users/mrcyrilgoud/Desktop/repos/Lab3/src/pipelines/lab3_pipeline_lib.py), focusing on preprocessing, normalization, augmentation, validation slice, checkpoint loading, and PSNR computation.

## What Matches

- Normalization is effectively aligned.
  - Baseline: `pil_to_tensor()` converts RGB to `float32` and divides by `255.0`.
  - SPAN: `load_rgb_tensor()` converts RGB to `uint8`, then to `float32`, then divides by `255.0`.
  - Both feed `[0, 1]` tensors into training and validation.

- PSNR math is also effectively aligned.
  - Baseline: `tensor_psnr()` computes per-image MSE on tensors clamped to `[0, 1]`, then averages per-image PSNR across the loader.
  - SPAN: `evaluate_psnr()` computes the same per-image PSNR on `[0, 1]` tensors, plus a secondary global-PSNR readout.
  - This is not enough to explain the observed gap.

## Ranked Causes

1. Validation-slice parity is broken in the recorded SPAN Modal run.
   - Current local data yields `input_psnr_first_16_lex = 21.595571064415275`.
   - The baseline `train64_val16` Modal run logged `input_psnr = 21.595572471618652`, which matches the current local first-16 lexicographic slice.
   - The recorded SPAN `train128_val16` Modal run logged `input_psnr = 21.038768760097977`.
   - The helper script checked every contiguous 16-image window in the current 100-image validation set; the closest one is `21.090915972649782`, still `0.05214721255180521` dB away.
   - Conclusion: the SPAN run was not scored on the same validation slice as the baseline path. Until that parity is fixed, the baseline-vs-SPAN PSNR comparison is not trustworthy.

2. Training preprocessing and augmentation do not match the baseline path.
   - Baseline training uses random `224x224` joint crops, horizontal mirror, vertical flip, and `k * 90` rotation.
   - SPAN training uses full `256x256` frames, horizontal flip, and `k * 90` rotation only. There is no random crop, and vertical flip is missing. `cutout_prob` exists but defaults to `0.0`.
   - Validation preprocessing is close on the current data because all local val images are already `256x256`, but the training distribution is materially different.
   - This changes both data diversity and the optimization target relative to the baseline search runs.

3. Checkpoint selection/evaluation is weaker than the baseline path.
   - Baseline training validates an EMA model every epoch, saves EMA weights to `best.pt`, then rebuilds a fresh model and reloads `best.pt` for final evaluation.
   - SPAN validates the live online model, saves the raw `state_dict`, and reports the in-loop best metrics directly. `_load_checkpoint()` is only used later for export.
   - This should be treated as a likely smaller regression/noise source rather than the primary cause of the large comparison gap above.

## Exact Path Differences

- Preprocessing
  - Baseline: crop-to-`224` during training, fit-to-`256` only when needed at eval.
  - SPAN: no crop; resize-to-`256` only when needed.

- Normalization
  - Baseline: RGB `float32 / 255.0`.
  - SPAN: RGB `uint8 -> float32 / 255.0`.
  - Net effect: same normalized range.

- Augmentation
  - Baseline: horizontal mirror + vertical flip + `0/90/180/270` rotation.
  - SPAN: horizontal flip + `0/90/180/270` rotation.

- Validation slice
  - Baseline comparable references are stable with the current repo data:
    - `train64_val16`: `input_psnr = 21.59557`
    - `train512_val100`: `input_psnr = 21.33616`
  - Recorded SPAN `train128_val16`: `input_psnr = 21.03877`
  - This is the strongest evidence of non-comparable evaluation.

- Checkpoint loading
  - Baseline: reloads best EMA checkpoint before final evaluation.
  - SPAN: saves raw best state during training; later export reload is separate from the training summary path.

- PSNR path
  - Both primary paths report mean per-image PSNR over `[0, 1]` tensors.
  - PSNR implementation is not the likely root cause.

## Highest-Confidence Fix To Try Next

Add a hard validation-parity gate to the SPAN Modal path, then rerun on the same comparison slice as the baseline.

Specifically: before trusting a SPAN Modal run, log `val_preview` and abort unless the measured LR-vs-HR baseline `input_psnr` matches the expected baseline value for the chosen slice (`21.59557` for the 16-image lex slice, `21.33616` for the 100-image lex slice on the current repo data). Once that gate passes, the next model-side adjustment to try is matching the baseline crop-based train preprocessing.
