# HR/LR Pairing Spec (Lab 3)

Canonical pairing pattern for all Lab 3 experiment notebooks. Applies to training data and validation data.

Reference implementation: [`U-Net Experiment 1/u_net_experiment_1_modal_run.py`](/Users/mrcyrilgoud/Desktop/repos/Lab3/U-Net%20Experiment%201/u_net_experiment_1_modal_run.py).

## Terminology

- **LR** — low-resolution image. Model **input**. `Data/LR_train/LR_train{i}/`, `Data/LR_val/`.
- **HR** — high-resolution image. Model **target**. `Data/HR_train/HR_train{i}/`, `Data/HR_val/`.
- **Pair** — `(LR, HR)` sharing the same file stem within the same split. For Lab 3, LR and HR have identical pixel dimensions; the "resolution" difference is the degradation applied to LR.

## Directory layout

```text
Data/
  HR_train/HR_train1..4/
  LR_train/LR_train1..4/
  HR_val/
  LR_val/
```

Train has four splits. Val has one. Pairing is by file stem, only within a split.

## The audit (the contract)

Before any training, run a pairing audit. A split passes only if all six hold:

1. Both HR and LR directories exist.
2. No duplicate stems in either HR or LR.
3. Every stem appears on both sides (no HR-only or LR-only).
4. Every pair opens as RGB via PIL.
5. `HR.size == LR.size` for every pair.
6. `paired_count > 0` for the split.

If any check fails, raise `RuntimeError` and refuse to train. Run the audit **twice per Modal launch**: once locally before upload, once remotely inside the Modal function before building DataLoaders.

For full runs (`run_mode='full'`), enforce total counts:

- `train_pairs_observed == EXPECTED_TRAIN_PAIRS` (`3036`)
- `val_pairs_observed == EXPECTED_VAL_PAIRS` (`100`)

Do not enforce expected totals for smoke/limited runs, but still enforce split-level integrity checks above.

## Required functions

Each conforming notebook must provide these helpers (required interface; current reference implementation is close but not yet fully aligned):

```python
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}
EXPECTED_TRAIN_PAIRS = 3036
EXPECTED_VAL_PAIRS = 100

def _file_maps(directory: Path) -> tuple[dict[str, Path], list[str]]:
    """Non-recursive. Returns (stem -> path, duplicate stems)."""

def audit_pair_directory(hr_dir: Path, lr_dir: Path, split_name: str) -> dict[str, Any]:
    """Runs the 6 split checks. Returns dict with counts, examples, and 'passed': bool.
    Raises FileNotFoundError if either directory is missing."""

def run_pairing_audit(data_root: Path, enforce_expected_totals: bool = True) -> dict[str, Any]:
    """Audits HR_train{1..4} + HR_val.
    Raises RuntimeError if any split fails.
    Also checks expected totals when enforce_expected_totals=True."""

def collect_train_pairs(data_root: Path, limit: int | None = None) -> list[tuple[Path, Path, str]]:
    """Returns (lr_path, hr_path, name) tuples. name = f'HR_train{i}/{stem}'."""

def collect_val_pairs(data_root: Path, limit: int | None = None) -> list[tuple[Path, Path, str]]:
    """Returns (lr_path, hr_path, name) tuples. name = stem."""
```

The tuple order `(lr, hr, name)` is fixed everywhere. Pairs lists are sorted by stem for determinism, and train split traversal order is fixed as `HR_train1`, `HR_train2`, `HR_train3`, `HR_train4`.

## Architecture: five rules (matching)

Goal: **the same semantic scene** in LR and HR stays aligned from files through tensors, for **train** (four `HR_train{i}` / `LR_train{i}` pairs) and **val** (`HR_val` / `LR_val`).

1. **Stem is the join key, inside one split only** — Build samples from the **intersection** of HR and LR stems in the paired folders (train: same `i` for `HR_train{i}` and `LR_train{i}`; val: `HR_val` and `LR_val` only). Never pair across `i`, never pair HR from one folder to LR from another, never zip two independent lists by row index.
2. **Non-recursive discovery** — List files with `Path.iterdir()` (flat directory). Do not use `rglob`; nested paths must not become extra “phantom” pairs.
3. **One index → one fixed `(lr_path, hr_path)` tuple** — The dataset is a list of tuples (sorted by stem for determinism). `__getitem__(i)` loads that tuple only. If you shuffle for training, shuffle **indices** or shuffle **one** list of tuples — never shuffle LR rows and HR rows separately.
4. **Shared geometry per sample** — Any crop, flip, or rotate (train) uses **one** RNG stream and applies the **same** box / decision to LR and HR. Val: no random crop or random flip; LR and HR stay aligned (optional fixed resize to eval size is allowed if applied the same way to both).
5. **Shared tensorization** — Same path from decoded image to tensor for LR and HR: e.g. `float32`, `CHW`, values in `[0, 1]`. Do not use different normalizations for LR vs HR.

**Train-only requirement (diversity, not join correctness)** — For train, include **epoch** in augmentation seeds (e.g. `set_epoch` + seed from `(base_seed, epoch, index)`), so the same index does not always see the same crop forever. This does not change *which* files are paired; it only varies the view.

## Modal volume handling

- **Always re-upload before a run.** Remove `/Data/HR_train`, `/Data/LR_train`, `/Data/HR_val`, `/Data/LR_val` from the volume, then upload fresh with `Volume.batch_upload(force=True)`. Never skip on "the directory already exists" — that's how you train on a half-populated volume from a prior aborted upload.
- **Use the Modal Python SDK** (`Volume.from_name`, `batch_upload`, `remove_file`) rather than shelling out to the `modal` CLI.
- **The remote audit catches volume corruption** that the local one can't see (truncated uploads, wrong mount path, etc.).

## What to record

In `summary.json` for every run:

- `pairing_audit`: the full audit dict from `run_pairing_audit` (remote).
- `pairing_audit.expected` and `pairing_audit.observed`: expected vs observed train/val totals.
- `pair_summary`: `{train_pairs, val_pairs, train_preview, val_preview}`.
- `config.train_pair_limit` / `config.val_pair_limit`: `null` for full runs, a number for smoke runs. Smoke runs must also set `run_mode = 'smoke'` so nobody reads a smoke PSNR as a full-run number.

## Failure matrix

| Scenario | Required behavior |
|---|---|
| Missing HR for a stem | Audit records `hr_only_count > 0`, raises |
| Corrupted image | Audit's RGB convert fails, recorded in `unreadable_pairs`, raises |
| `HR.size != LR.size` | Audit records `size_mismatches`, raises |
| Duplicate stem in HR dir | Recorded in `duplicate_hr_basenames`, raises |
| Missing `HR_val/` | `FileNotFoundError` raised immediately |
| Split has zero matched pairs | Audit records `paired_count == 0`, raises |
| Full run has `train_pairs_observed != 3036` or `val_pairs_observed != 100` | Audit raises before DataLoader creation |
| Stray subdir file (`HR_train1/backup/foo.png`) | Silently ignored (non-recursive discovery) |
| Stale Modal volume | Prevented by fresh re-upload; also caught by remote audit |
| Crop box differs between LR and HR | Forbidden — both use the same box derived from LR size |
| Same crop box used for a sample every epoch | Forbidden — train RNG must depend on epoch |
| LR list and HR list shuffled independently | Forbidden — shuffle one list of `(lr, hr, …)` tuples or shuffle indices only |

## Current status

- **U-Net Experiment 1**: Matches file-level LR–HR alignment rules; still needs full alignment for (a) epoch-aware train augmentation seeding and (b) strict audit gating for zero-pair splits / full-run expected totals.
- **SPAN NPU v1**: Does not follow this spec (`rglob`, no audit, no fresh re-upload, shells out to `modal` CLI).
- **Restormer NPU v1**, **Teacher-Student Reformer**: not yet reviewed.

## Related

- [`docs/model_notebook_requirements_rubric.md`](/Users/mrcyrilgoud/Desktop/repos/Lab3/docs/model_notebook_requirements_rubric.md) — overall Lab 3 notebook requirements.
- [`U-Net Experiment 1/u_net_experiment_1_modal_run.py`](/Users/mrcyrilgoud/Desktop/repos/Lab3/U-Net%20Experiment%201/u_net_experiment_1_modal_run.py) — reference implementation (pairing + audit + tuple list; pending full alignment with the hard gates in this spec).
