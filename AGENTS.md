# Repository Guidelines

## Project Structure & Module Organization
Core training and inference scripts live at the repository root: `train.py`, `isolated_nwm_infer.py`, `isolated_nwm_eval.py`, and `planning_eval.py`. Model and diffusion internals are split into reusable modules: `models.py`, `datasets.py`, `distributed.py`, and `diffusion/` (noise schedules, samplers, and utilities). Runtime configs are in `config/*.yaml`. Dataset split metadata is in `data_splits/<dataset>/{train,test}/`. Notebooks (`interactive_model.ipynb`, `visualize_*.ipynb`) are for exploration, not production runs.

## Build, Test, and Development Commands
- `mamba env create -f env.yaml && mamba activate nwm`: create the baseline environment.
- `python train.py --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300 --torch-compile 0`: single-node training/debug.
- `torchrun --nproc-per-node=8 train.py --config config/nwm_cdit_xl.yaml ...`: distributed training.
- `python isolated_nwm_infer.py --exp config/nwm_cdit_xl.yaml --datasets recon --eval_type time --output_dir $RESULTS_FOLDER --gt 1`: generate GT frames for eval.
- `python isolated_nwm_eval.py --datasets recon --gt_dir $RESULTS_FOLDER/gt --exp_dir $RESULTS_FOLDER/nwm_cdit_xl --eval_types time`: compute metrics.
- `torchrun --nproc-per-node=8 planning_eval.py --exp config/nwm_cdit_xl.yaml --datasets recon ...`: planning evaluation.

## Coding Style & Naming Conventions
Use Python 3.10 with 4-space indentation and PEP 8 naming (`snake_case` functions/variables, `PascalCase` classes). Keep module-level responsibilities focused (data loading in `datasets.py`, modeling in `models.py`). Prefer explicit config keys in YAML and avoid hard-coded paths; use flags such as `--config` and `--output_dir`.

## Testing Guidelines
There is no dedicated unit-test suite in-tree. Validate changes with targeted smoke runs:
- training path: short run of `train.py` on one GPU with reduced epochs/steps.
- inference/eval path: one dataset (`recon`) through `isolated_nwm_infer.py` and `isolated_nwm_eval.py`.
Name any new tests/scripts by behavior, e.g. `test_<feature>.py` or `<feature>_smoke.sh`.

## Commit & Pull Request Guidelines
History favors short, imperative subjects with optional prefixes: `feat:`, `fix:`, `docs:` (example: `fix: zero_grad bug`). Keep commits scoped to one logical change. For PRs, follow `CONTRIBUTING.md`: include a clear description, reproduction/validation steps, linked issue(s), API/docs updates when relevant, and confirm tests/lint pass. Complete the Meta CLA before first contribution.
