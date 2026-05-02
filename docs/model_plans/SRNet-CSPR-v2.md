# SRNet CSPR v2 — Cascaded Sub-Pixel Refinement Plan

## Summary
- New notebook at [lab3_srnet_npu_v2_cspr_modal.ipynb](/Users/cyrilgoud/Desktop/repos/personal/Lab3/experiments/SRNet%20NPU%20v2_CSPR/lab3_srnet_npu_v2_cspr_modal.ipynb).
- Keeps the Lab 3 detached Modal workflow, L3 dataset routing, `256x256x3 -> 256x256x3` contract, and `.pth -> .onnx -> calibration -> conversion script -> .mxq` chain.
- Replaces the U-Net + Pre-Dec2 student with a **4-stage feature-space cascade** with sub-pixel upsampling and a single trailing global residual `Add`.
- Authorized relaxation: the MLA PDF lists `Add` as *CPU Fallback* (not Failed); the audit allows `Add` but caps the count at exactly 1.
- Target: `best_val_psnr >= 24.30 dB`.

## Architecture Changes
- New model class: `SRNetCSPRv2`.
- Forward graph contains only:
  - `Conv2d`
  - `LeakyReLU`
  - `PixelShuffle` (exports as `DepthToSpace`)
  - exactly one trailing `Add` (`output = LR + delta`)
- Building blocks:
  - `DSConvBlock(C, k)`: depthwise `Conv(C->C, k, groups=C)` + LeakyReLU + pointwise `Conv(C->C, k=1)` + LeakyReLU. No skip add.
  - `DownConv(Cin, Cout)`: `Conv3x3 stride=2` + LeakyReLU.
  - `SubPixelUp(Cin, Cout)`: `Conv(Cin -> Cout*4, k=3)` + `PixelShuffle(2)` + LeakyReLU.
  - `FeatureRefineStage(C=64, Cb=112)`: `DSConvBlock(C,3) x 2 -> DownConv(C->Cb) -> DSConvBlock(Cb,5) x 2 -> DSConvBlock(Cb,3) -> SubPixelUp(Cb->C) -> DSConvBlock(C,3)`.
- Full network:
  - stem: `Conv3x3 3->64` + LeakyReLU
  - `FeatureRefineStage(64, 112)` x 4
  - tail_head: `DSConvBlock(64, 3)`
  - tail: `Conv3x3 64->3` (small-init, scale 1e-3)
  - output: `LR + tail(...)` — the single trailing Add
- Training-only deep supervision via `forward_with_aux`:
  - `aux_proj`: 3 `Conv3x3 64->3` heads (one per stage 1, 2, 3)
  - aux predictions are `LR + aux_proj_k(stage_k_features)`
  - **never traversed in `forward`**, so they stay out of the export graph
- Total params ≈ 1.54 M, leaf modules: `Conv2d`, `LeakyReLU`, `PixelShuffle` (verified locally).

## Training Loss
- Supervised residual loss unchanged from v1:
  - `pred_res = pred - lr`, `target_res = hr - lr`
  - `loss_mse = MSE(pred_res, target_res)`, `loss_l1 = L1(pred_res, target_res)`
- Teacher distillation unchanged from v1:
  - `loss_distill = MSE(pred_res, teacher_res.detach())`
- New deep-supervision term:
  - `loss_ds = mean over k of MSE(aux_pred_k - lr, target_res)`
- Total loss:
  - `compute_supervised_loss + cfg.distill_weight * loss_distill + cfg.ds_weight * loss_ds`
  - defaults: supervised weighting `0.8 MSE + 0.2 L1`, `distill_weight = 0.05`, `ds_weight = 0.10`

## Teacher
- FSRCNN 96/40/m8 teacher unchanged from v1:
  - default checkpoint resolution order:
    1. [best.pth](/Users/cyrilgoud/Desktop/repos/personal/Lab3/runs/013420_2904_fsrcnn_residual_96_40_m8_modal_resume5750_to6000_lr300/checkpoints/best.pth)
    2. [best.pth](/Users/cyrilgoud/Desktop/repos/personal/Lab3/runs/021424_2904_fsrcnn_residual_96_40_m8_modal_resume5750_to6000_lr300/checkpoints/best.pth)
  - teacher weight source default: `ema`
  - if neither checkpoint exists, distillation is silently disabled and recorded

## Warm Start
- Disabled by default: `cfg.warm_start_checkpoint = ""`.
- The CSPR parameter names do not overlap the v1 U-Net layout, so filtered same-name same-shape loading would yield zero compatible keys. From-scratch is cleaner and avoids noisy report state.
- Filtered loader is preserved; setting `LAB3_SRNET_WARM_START_CHECKPOINT` re-enables it for any compatible future variant.

## Training Defaults
- `batch_size = 8`
- `epochs = 460`
- `warmup_epochs = 8`
- `lr = 2e-4`, `min_lr = 5e-6`
- optimizer: `AdamW(betas=(0.9, 0.99), weight_decay=5e-5)`
- `grad_clip_norm = 0.5`
- `use_amp = True`, `ema_decay = 0.999`
- `early_stop_patience = 50`
- `cspr_channels = 64`, `cspr_bottleneck_channels = 112`, `cspr_num_stages = 4`
- `ds_weight = 0.10`

## ONNX Audit
- Tightened policy:
  - `SAFE_LEAF_MODULES = {Conv2d, LeakyReLU, PixelShuffle}`
  - `PDF_ALLOWED_ONNX_OPS = {Conv, LeakyRelu, DepthToSpace, Add}`
  - `FORBIDDEN_ONNX_OPS = {Concat, Clip, Mul, Sub, Relu, Resize, MatMul, Softmax, Sigmoid, Transpose, Tanh, SpaceToDepth}`
  - new gate: `add_count <= PDF_ADD_LIMIT` (PDF_ADD_LIMIT = 1)
- Export bumped to `opset_version = 17`.
- Strict and legacy fail conditions both include `not graph_audit["add_within_limit"]`.
- ORT parity check unchanged (`max_diff < 1e-3` against PyTorch CPU forward).

## Reporting
- `validation_summary`: metrics only — `val_psnr`, `input_psnr`, `delta_psnr`. No baked-in baseline comparisons.
- `onnx_summary` extends with: `add_count`, `add_limit`, `add_within_limit`.
- `promotion_gates`: operational gates only — `contract_pass`, `safe_ops_pass`, `onnx_pass`, `calibration_pass`, `mxq_handoff_pass`, `promotion_pass`. PSNR-vs-baseline judgments are the operator's job.

## Files
- New artifacts location:
  - notebook: [experiments/SRNet NPU v2_CSPR/lab3_srnet_npu_v2_cspr_modal.ipynb](/Users/cyrilgoud/Desktop/repos/personal/Lab3/experiments/SRNet%20NPU%20v2_CSPR/lab3_srnet_npu_v2_cspr_modal.ipynb)
  - run output template: `runs/<RUN_DAY>/lab3_srnet_cspr_v2_<TS>/...`
  - conversion script written by the run: `convert_srnet_cspr_v2_mxq.py`
  - latency script written by the run: `measure_srnet_cspr_v2_npu_latency.py`
  - Modal app name: `lab3-srnet-cspr-v2`
  - runtime module path on remote: `/root/srnet_cspr_v2_modal_entry.py`

## Test Plan
- Static checks (verified locally):
  - notebook JSON parses
  - all code cells compile
  - contract `1x3x256x256 -> 1x3x256x256`
  - leaf-module audit contains only `Conv2d`, `LeakyReLU`, `PixelShuffle`
  - `forward_with_aux` returns 3 aux predictions for `num_stages = 4`
  - param count ≈ 1.54 M
- Modal run checks:
  - L3 pair counts remain `2217` train and `110` val
  - training, export, and calibration all run on Modal only
  - sync-back creates the day-partitioned local run directory
  - ONNX export op set ⊆ `{Conv, LeakyRelu, DepthToSpace, Add}`
  - `onnx_summary.add_count == 1` and `add_within_limit == True`
  - `onnx_summary.pdf_allowed_ops_only == True`
  - `onnx.checker` and ORT parity pass
  - calibration manifest is training-derived, count = 128
- Acceptance criteria:
  - strict MLA-safe ONNX audit passes under the updated v2 policy (one trailing Add allowed)
  - `best_val_psnr > 24.0179` to beat the current SRNet recovery
  - target `best_val_psnr >= 24.0634` (beat FSRCNN reference)
  - stretch `best_val_psnr >= 24.1335` (beat FSRCNN @ 8000 epochs)
  - user target `best_val_psnr >= 24.30`
  - if MLA-safe export passes but PSNR regresses below `24.0179`, classify as a latency-oriented architectural experiment, not the new primary SRNet

## Assumptions
- `Add` is acceptable on Mobilint MLA-100 as long as it appears at most once in the exported graph (CPU Fallback per the PDF; see [bence.pasuko.com_mla-supported-operations_.pdf](/Users/cyrilgoud/Desktop/repos/personal/Lab3/docs/references/bence.pasuko.com_mla-supported-operations_.pdf)).
- `PixelShuffle` exports as `DepthToSpace`, which the PDF marks as supported.
- The FSRCNN teacher may contain unsupported inference ops in its own exported graph, but that is acceptable because the teacher is training-only and not part of the submission model.
- No local training, no autopilot controller changes, and no canonical registry changes are part of this patch.
