# Autopilot Setup Notes

## What Was Created

- Updated [AGENTS.md](/Users/mrcyrilgoud/Desktop/repos/Lab3/AGENTS.md) with explicit Lab 3 autopilot rules.
- Added the repo-local skill under [skills/lab3-autopilot](/Users/mrcyrilgoud/Desktop/repos/Lab3/skills/lab3-autopilot).
- Added the starter controller at [scripts/autopilot_controller.py](/Users/mrcyrilgoud/Desktop/repos/Lab3/scripts/autopilot_controller.py).
- Added the validator at [scripts/validate_canonical_pipeline.py](/Users/mrcyrilgoud/Desktop/repos/Lab3/scripts/validate_canonical_pipeline.py).
- Added the canonical-notebook launcher at [scripts/run_modal_experiment.py](/Users/mrcyrilgoud/Desktop/repos/Lab3/scripts/run_modal_experiment.py).
- Added report starters under [runs/autopilot_reports](/Users/mrcyrilgoud/Desktop/repos/Lab3/runs/autopilot_reports).

## Main Controller

The main starter controller is [scripts/autopilot_controller.py](/Users/mrcyrilgoud/Desktop/repos/Lab3/scripts/autopilot_controller.py).

Its current job is to:

- validate the canonical notebook first
- bootstrap normalized history from existing Lab 3 run summaries
- avoid exact duplicate configs unless a rerun reason is provided
- select the next candidate from ledger history and launch a bounded real Modal run through the canonical notebook path in `scripts/run_modal_experiment.py`
- update `ledger.jsonl`, `best_known.json`, and `inbox_summary.md`

## Canonical Notebook In The Loop

The canonical Modal entrypoint remains [lab3_wide_residual_nobn_modal_app.ipynb](/Users/mrcyrilgoud/Desktop/repos/Lab3/lab3_wide_residual_nobn_modal_app.ipynb).

The scaffold does not replace it. The controller now validates and launches through that notebook path before autonomous experimentation.

## What Is Still TODO

- Improve candidate mutation policy beyond the current nearest-neighbor ledger-driven selector.
- Add richer failure triage for interrupted Modal runs and explicit remote cancellation handling.
- Add deeper notebook validation checks if you want more than the current bounded end-to-end smoke run.

## Testing

Use this sequence to test that the automation works and is able to make progress:

```bash
python3 scripts/validate_canonical_pipeline.py \
  --notebook /Users/mrcyrilgoud/Desktop/repos/Lab3/lab3_wide_residual_nobn_modal_app.ipynb

python3 scripts/autopilot_controller.py \
  --max-runs 1 \
  --force-candidate wide_residual_nobn_v1 \
  --rerun-reason "real Modal smoke test" \
  --train-pairs 8 \
  --val-pairs 4 \
  --num-epochs 1 \
  --warmup-epochs 1 \
  --budget-minutes-per-run 10
```

Expected proof of progress:

- canonical notebook validation passes
- `runs/autopilot_reports/ledger.jsonl` exists and contains normalized entries
- `runs/autopilot_reports/best_known.json` exists
- `runs/autopilot_reports/inbox_summary.md` is refreshed
- at least one new real Modal run is recorded when the controller launches a bounded smoke candidate
- the latest ledger entry includes validation metrics plus ONNX, calibration, and MXQ handoff fields when the run completes

## What Must Be Wired Next

- Add smarter architecture mutations and promotion heuristics on top of the real synced ledger.
- Decide whether the controller should actively cancel remote Modal work when the outer wall-clock budget is exceeded.
