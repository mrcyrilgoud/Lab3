# Restormer teacher (Teacher–Student / distillation path)

## Role

This **Restormer-style teacher** is **not** the Lab 3 NPU submission. It is an accuracy-first model (LayerNorm, attention, GELU, etc.) used to generate **soft targets** and **teacher residuals** for a later **student** model that respects NPU / MXQ constraints.

- **I/O contract**: LR and HR are both **256×256×3** (same-resolution restoration).
- **Canonical implementation**: [`restormer_teacher/model.py`](../restormer_teacher/model.py) inside the Python package is the only supported teacher-model source of truth.
- **Primary training path (recommended)**: **Modal** using [`notebooks/restormer_teacher_modal_app.ipynb`](../notebooks/restormer_teacher_modal_app.ipynb) and [`tools/restormer_teacher_modal_app.py`](../tools/restormer_teacher_modal_app.py), with datasets on the **`lab3-data`** volume and artifacts on **`lab3-runs`** (same pattern as the canonical Lab 3 notebook).
- **Local path**: [`scripts/train_restormer_teacher.py`](../scripts/train_restormer_teacher.py) for smoke tests and debugging on your machine.

## Compatibility and retraining

- The current teacher checkpoints carry explicit `teacher_model_version` and `config_fingerprint` metadata.
- **Pre-fix / unversioned checkpoints are legacy** because earlier runs used an incorrect bias-free LayerNorm implementation.
- Legacy checkpoints may still be loaded for inspection or resume troubleshooting, but they are **not valid distillation sources**.
- Regenerate teacher targets only from current-version checkpoints. If you need inspection-only outputs from a legacy checkpoint, use `--allow-legacy-checkpoint` explicitly.

## Artifacts per run

Under `runs/<YYYY-MM-DD>/<run_name>/` (or your configured `artifact_root`):

| File | Purpose |
|------|---------|
| `metrics.jsonl` | One line per epoch (and optional `train_step` lines); PSNR, losses, deltas. |
| `history.json` | Cumulative `meta` + `epochs[]` for plotting and comparing runs. |
| `latest_status.json` | Small snapshot for dashboards / agents. |
| `run_config.json` | Resolved config used for the run. |
| `checkpoints/best_ema.pth` | Best validation PSNR (EMA weights). |
| `checkpoints/latest.pth` | Resume-friendly latest state. |
| `val_samples/epoch_*/` | LR, HR, EMA prediction, amplified residual PNGs. |

## Target generation

After training, run [`scripts/generate_teacher_targets.py`](../scripts/generate_teacher_targets.py) with `--checkpoint` pointing at `best_ema.pth` and `--output-dir` such as `runs/restormer_teacher/<run_id>/teacher_targets`.

- **`metadata.jsonl`**: one JSON object per training image (`identity_psnr`, `teacher_psnr`, `teacher_delta_psnr`, `use_for_distillation`).
- Current target metadata also records `teacher_model_version` and `config_fingerprint`.
- **`--only-save-improved`**: skips writing PNG/NPY when the teacher does **not** beat LR identity; metadata lines are still appended for a full audit trail.
- **`--save-residuals`**: writes `teacher_pred - lr` as `.npy` (and an amplified PNG).
- **Legacy checkpoints**: rejected by default for target generation.

## Student / distillation handoff

Use three spatial signals where available:

1. Ground-truth **HR**
2. **Teacher prediction** (same shape as LR/HR)
3. **Teacher residual** `teacher_pred - lr`

**Trust rule**: only treat teacher pixels or samples as supervision where **`teacher_psnr > identity_psnr`** (see `use_for_distillation` in metadata).

### Suggested student loss (starting point)

```
0.50 * L1(student, HR)
+ 0.30 * L1(student, teacher_pred)
+ 0.10 * L1(student_residual, teacher_residual)
+ 0.10 * (edge or frequency loss vs HR)
```

Tune weights after you measure student capacity and latency targets.

## Modal quick launch

From the Lab3 repo root (with Modal CLI authenticated):

```bash
python "Teacher-Student Reformer/tools/restormer_teacher_modal_app.py" \
  --config "Teacher-Student Reformer/configs/restormer_teacher.yaml" \
  --run-name restormer_teacher_large \
  --sync-data
```

Add `--smoke-test` for a short architecture smoke on the remote GPU. After completion, artifacts appear under `runs/<started-day>/<run_name>/` locally once the runs volume is synced.

If a Modal run times out before creating a remote run directory, the helper now returns a structured timeout result instead of masking the cutoff with a failed volume sync.

## Environment variables (notebook)

Typical overrides (see the Modal notebook cells):

- `LAB3_MODAL_GPU` — e.g. `L40S`
- `LAB3_NOTEBOOK_MODAL_DATA_VOLUME` — default `lab3-data`
- `LAB3_NOTEBOOK_MODAL_RUNS_VOLUME` — default `lab3-runs`
- `MODAL_RUN_DAY` — optional `YYYY-MM-DD` override for the run folder

## Path normalization

If JSON artifacts contain `/mnt/lab3-data` or `/mnt/lab3-runs` paths after a Modal run, replace them with your local `Data/` and `runs/` roots when analyzing offline (similar idea to `normalize_synced_run` in the main Lab3 Modal helper).
