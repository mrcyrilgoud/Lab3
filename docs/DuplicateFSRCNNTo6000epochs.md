# Duplicate FSRCNN Residual Notebook To 6000

## Goal
Create a clean duplicate of the 5750 Colab notebook at:
`experiments/FSRCNNResidual/lab3_fsrcnn_residual_colab_resume_6000_most_recent_last_lr300_restart.ipynb`

Use as source:
`experiments/FSRCNNResidual/lab3_fsrcnn_residual_colab_resume_5750_most_recent_last_lr150_restart.ipynb`

Do not modify the source notebook.

## Rubric Alignment
Implementation must stay aligned with:
`docs/notes/model_notebook_requirements_rubric.md`

For this task, only duplication/config updates/sanitization are required; no local training run.

## Required Config Edits (RunConfig cell)
Apply these exact values in the duplicated notebook:

```python
@dataclass
class RunConfig:
    model_id: str = 'fsrcnn_residual_96_40_m8'
    run_slug: str = 'fsrcnn_residual_96_40_m8_resume6000_most_recent_last_lr300_restart'
    enable_resume: bool = True
    resume_checkpoint_path: str | None = (
        '/content/drive/MyDrive/Data 255 Class Spring 2026/Data 255/Lab 3/'
        'runs/220035_2804_fsrcnn_residual_96_40_m8_resume5750_most_recent_last_lr150_restart/checkpoints/last.pth'
    )

    epochs: int = 6000
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    load_optimizer_state: bool = False
```

Also rename title/header text in markdown/config labels from:
- `resume_5750 ... lr150`
to:
- `resume_6000 ... lr300`

## Implementation Steps
1. Copy the 5750 source notebook to the new 6000 target path.
2. Update title/header text and `RunConfig` values in the duplicated notebook.
3. Clear all code-cell outputs and execution counts in the duplicated notebook.
4. Save and verify the resulting notebook is valid JSON.
5. Run light static checks on the duplicated notebook only.

## Light Validation Checks
Confirm in the new notebook:
- `epochs: int = 6000`
- `lr: float = 3e-4`
- `run_slug: str = 'fsrcnn_residual_96_40_m8_resume6000_most_recent_last_lr300_restart'`
- resume path contains:
  `220035_2804_fsrcnn_residual_96_40_m8_resume5750_most_recent_last_lr150_restart/checkpoints/last.pth`
- no old `resume5400` / `182012_2804...resume5400...` reference remains
- all code cells have cleared outputs and null execution counts

Example verification commands:

```bash
python -m json.tool experiments/FSRCNNResidual/lab3_fsrcnn_residual_colab_resume_6000_most_recent_last_lr300_restart.ipynb >/dev/null
rg "epochs: int = 6000|lr: float = 3e-4|run_slug: str = 'fsrcnn_residual_96_40_m8_resume6000_most_recent_last_lr300_restart'" experiments/FSRCNNResidual/lab3_fsrcnn_residual_colab_resume_6000_most_recent_last_lr300_restart.ipynb
rg "220035_2804_fsrcnn_residual_96_40_m8_resume5750_most_recent_last_lr150_restart/checkpoints/last\\.pth" experiments/FSRCNNResidual/lab3_fsrcnn_residual_colab_resume_6000_most_recent_last_lr300_restart.ipynb
rg "resume5400|182012_2804_fsrcnn_residual_96_40_m8_resume5400" experiments/FSRCNNResidual/lab3_fsrcnn_residual_colab_resume_6000_most_recent_last_lr300_restart.ipynb
```

## Assumptions
- “Most recent run checkpoint” means 5750 run `last.pth` (not `best.pth`).
- Notebook remains Colab + Google Drive based.
- No notebook execution is performed as part of this task.
