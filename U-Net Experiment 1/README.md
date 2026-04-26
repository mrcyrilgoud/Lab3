# U-Net Experiment 1

This folder contains a standalone Lab 3 notebook:

- [u_net_experiment_1_modal_app.ipynb](/Users/mrcyrilgoud/Desktop/repos/Lab3/U-Net%20Experiment%201/u_net_experiment_1_modal_app.ipynb)

The notebook is self-contained. It does not import repo-local helper modules for training, export, calibration, or Modal execution.

## Model Overview

The model is `SingleBridgeUNet`, a small U-Net-style super-resolution model with a residual output:

- input: `1 x 3 x 256 x 256`
- output: `1 x 3 x 256 x 256`
- default parameter count: about `7.7M`
- final prediction: `output = input + delta`

### Architecture

- Stem: `Conv2d(3 -> 56)` + `LeakyReLU`
- Encoder:
  - `enc1` at full resolution
  - `enc2` after stride-2 downsample
  - `enc3` after another stride-2 downsample
  - bottleneck after a third stride-2 downsample
- Decoder:
  - `ConvTranspose2d` upsampling
  - local decoder refinement blocks after each upsample
- Skip connections:
  - one always-on shallow additive bridge at the highest decoder resolution
  - optional deeper additive bridges controlled by `use_deep_skip_bridges`
- Tail:
  - `Conv2d(... -> 3)` with small initialization

### Why it is structured this way

- The model keeps the Lab 3 contract fixed at `256x256x3`.
- It avoids batch normalization and concatenation.
- It keeps the forward path mostly to `Conv`, `ConvTranspose`, `LeakyReLU`, and `Add`.
- It predicts a residual correction instead of a full image from scratch.

## Training and Validation

The notebook uses paired LR/HR data from:

- `Data/HR_train/HR_train1..4`
- `Data/LR_train/LR_train1..4`
- `Data/HR_val`
- `Data/LR_val`

Before any Modal launch, it performs a hard pairing audit:

- verifies all expected folders exist
- matches HR/LR files by basename
- checks for duplicate basenames
- checks every paired image for RGB readability
- checks every paired image for size equality
- fails immediately on any mismatch

Training details:

- loss: residual L1 loss
- optimizer: `AdamW`
- schedule: warmup + cosine decay
- EMA model used for evaluation and checkpointing
- validation reports:
  - `val_psnr`
  - `input_psnr`
  - `delta_psnr`

## Export and Submission Artifacts

The notebook produces the full Lab 3 artifact chain except actual MXQ compilation:

- checkpoint: `checkpoints/best.pt`
- ONNX: `exports/best.onnx`
- calibration set: `exports/calibration/`
- calibration manifest derived from training data only
- MXQ handoff metadata in the final report

ONNX checks include:

- ONNX checker pass
- ONNX Runtime parity check
- ONNX op audit against a set of known supported or CPU-fallback ops from the MLA notes

## Notebook Structure

The notebook is organized around this flow:

1. Setup and imports
2. Pairing audit and config printout
3. Model, dataset, training, export, and calibration code
4. Experiment config dataclass
5. Remote pipeline execution function
6. Modal app, volumes, upload, launch, and sync-back
7. Optional smoke gate before full run

Important design choice:

- local execution is for preflight, upload, launch, and sync-back
- training, validation, export, and calibration run on Modal only

## Running in Modal

The notebook expects:

- Modal Python SDK installed locally
- authenticated Modal profile
- access to volumes:
  - `lab3-data`
  - `lab3-runs`

The Modal app is defined inline in the notebook:

- app name: `lab3-unet-experiment-1`
- function: `run_unet_experiment_1`

### Important run behavior

- The notebook refreshes the required dataset folders in `lab3-data` before a run:
  - `HR_train`
  - `LR_train`
  - `HR_val`
  - `LR_val`
- It does not trust an already non-empty `/Data` volume.
- It syncs the finished run back from `lab3-runs` into local `runs/<day>/<run_name>/`.

### Key environment variables

- `UNET_EXPERIMENT_EXECUTE_MODAL=true`
- `UNET_EXPERIMENT_RUN_MODE=smoke|full`
- `UNET_EXPERIMENT_MODAL_GPU=A10G|L40S`
- `UNET_EXPERIMENT_SYNC_DATA=true`
- `UNET_EXPERIMENT_FORCE_DATA_SYNC=false|true`

Useful defaults:

- smoke run:
  - `1` epoch
  - `8` train pairs
  - `4` val pairs
- full run:
  - `34` epochs
  - full paired training set
  - full validation set

## Modal Gotchas

- Use the notebook or the embedded launcher generated from it. Do not use the old launcher that reads the notebook from a host-only absolute path inside the container.
- The notebook sets the Modal image Python version from the local notebook kernel major/minor version. This matters for notebook-defined Modal functions.
- A stopped app after completion is normal. A live app with no logs is usually a provisioning issue, not a notebook logic issue.

## Outputs to Check

After a successful run, inspect:

- `run_config.json`
- `metrics.jsonl`
- `latest_status.json`
- `summary.json`
- `report.json`

Typical local output path:

- [runs/2026-04-16/u_net_experiment_1_20260416_235649](/Users/mrcyrilgoud/Desktop/repos/Lab3/runs/2026-04-16/u_net_experiment_1_20260416_235649)

## Current Known Result

The notebook has already completed a successful smoke run on Modal:

- GPU: `A10G`
- mode: `smoke`
- best validation PSNR: about `23.021`
- ONNX checker: passed
- ORT parity: passed
- calibration export: passed

That confirms the notebook is operational end-to-end through training, ONNX export, calibration generation, and artifact sync-back.
