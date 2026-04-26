# Lab 3 Autopilot Skill

This skill is the repo-local operating guide for Codex automation in this Lab 3 repository.

Use it when the task involves:

- bounded autonomous experiment planning
- validating the canonical Modal notebook before search
- ranking same-slice, same-budget candidates
- avoiding duplicate reruns
- updating local autopilot reports for handoff

The skill does not replace the rubric. It assumes the rubric is read first and keeps the canonical entrypoint fixed at [lab3_wide_residual_nobn_modal_app.ipynb](/Users/mrcyrilgoud/Desktop/repos/Lab3/lab3_wide_residual_nobn_modal_app.ipynb).

## How To Use It

- Read [skills/lab3-autopilot/SKILL.md](/Users/mrcyrilgoud/Desktop/repos/Lab3/skills/lab3-autopilot/SKILL.md).
- Validate the canonical notebook.
- Run [scripts/autopilot_controller.py](/Users/mrcyrilgoud/Desktop/repos/Lab3/scripts/autopilot_controller.py) to bootstrap history, avoid duplicate configs, launch a real bounded Modal run, and record the next experiment step.
- Keep all automation-facing status in [runs/autopilot_reports](/Users/mrcyrilgoud/Desktop/repos/Lab3/runs/autopilot_reports).

## Testing

To test that the automation works and is able to make progress:

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

Progress is proven when:

- the canonical notebook validation passes
- `runs/autopilot_reports/ledger.jsonl` gains normalized history or a new real run entry
- `runs/autopilot_reports/best_known.json` is present and updated
- `runs/autopilot_reports/inbox_summary.md` is refreshed
- the latest ledger entry includes real metrics and synced ONNX/calibration/MXQ handoff fields when the run completes
