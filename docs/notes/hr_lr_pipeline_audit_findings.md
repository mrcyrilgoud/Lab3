# HR/LR Pipeline Audit Findings

Audit command:

```bash
python3 src/scripts/audit_lab3_data_pipeline.py --json-out tmp/hr_lr_pipeline_audit.json
```

Result on the current repo state: the on-disk Lab 3 dataset looks clean, so the main issue does **not** look like an HR/LR pairing failure.

## What passed

- `Data/HR_train{1..4}` and `Data/LR_train{1..4}` pair cleanly at the file level.
- `Data/HR_val` and `Data/LR_val` pair cleanly at the file level.
- Observed totals match the full-run contract exactly: `3036` train pairs and `100` val pairs.
- No duplicate stems inside a split, no HR-only or LR-only files, no unreadable pairs, and no `HR.size != LR.size` mismatches.
- All observed pairs are already `256x256`, so the current validation loader does not need to resize the present dataset.

## What the audit found

- The strongest signal is a **train/val difficulty gap**, not broken pairing.
  Train LR->HR baseline PSNR is `26.112-27.078 dB` across train splits, while val is `21.336 dB`.
  The weighted train-vs-val baseline gap is `5.421 dB`, which is large enough to explain a lot of apparent PSNR drop by itself.
- The current shared pipeline does **not** enforce the pairing audit before building loaders.
  [`run_pipeline`](/Users/mrcyrilgoud/Desktop/repos/Lab3/src/pipelines/lab3_pipeline_lib.py:1150) calls `collect_train_pairs` / `collect_val_pairs` directly and never gates on `run_pairing_audit`.
- Train augmentation is synchronized between LR and HR, but it is **not epoch-aware**.
  [`PairedSRDataset.__getitem__`](/Users/mrcyrilgoud/Desktop/repos/Lab3/src/pipelines/lab3_pipeline_lib.py:443) seeds train sampling with `self.seed + index`, so the same sample sees the same crop/flip/rotation every epoch.
- Limited runs are **slice-biased**.
  `train64` and `train256` come entirely from `HR_train1`, and `train1024` is only `HR_train1` plus part of `HR_train2`.
  That makes smoke or bounded runs poor proxies for full-run behavior.
- Naming is safe in the current tuple list because train names include the split prefix (`HR_train{i}/stem`), but stems are heavily reused across splits (`782` reused train stems), so any future code that drops the split prefix or pairs across folders will break badly.
- The notebook/launcher path has drifted after the repo move.
  [`src/scripts/run_modal_experiment.py`](/Users/mrcyrilgoud/Desktop/repos/Lab3/src/scripts/run_modal_experiment.py:16) still expects `src/lab3_wide_residual_nobn_modal_app.ipynb`, which does not exist; the notebook that does exist is [experiments/00_baseline/lab3_wide_residual_nobn_modal_app.ipynb](/Users/mrcyrilgoud/Desktop/repos/Lab3/experiments/00_baseline/lab3_wide_residual_nobn_modal_app.ipynb).
- Modal data sync still allows stale-volume reuse.
  [`sync_data_volume`](/Users/mrcyrilgoud/Desktop/repos/Lab3/src/scripts/lab3_modal_app.py:66) skips upload when `/Data` already exists, which conflicts with the pairing spec's "always re-upload" rule.

## Bottom line

Most likely root cause: **something other than raw data pairing**, led by the train/val distribution gap and reinforced by current pipeline guardrail gaps (`no enforced audit`, `no epoch-aware crop seed`, `biased limited slices`, `stale notebook/sync assumptions`).

Least likely root cause: a present-day HR/LR file mismatch in `Data/`.
