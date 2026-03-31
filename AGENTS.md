# Repository Guidelines

## Project Structure & Module Organization
Core training and evaluation entry points live under `scripts/`: `scripts/train.py`, `scripts/plan_eval.py`, `scripts/infer.py`, and `scripts/evaluate.py`. Model and runtime internals live under `src/` with direct imports from `src.core`, `src.data`, `src.models`, `src.diffusion`, and related packages. Configuration lives in `configs/**/*.yaml`; keep new experiment variants under `configs/experiment/` and follow the existing `nwm_cdit_*.yaml` naming pattern. Helper scripts live in `scripts/`, while static split metadata lives in `data/splits/`. Generated plots and notebook experiments belong in `artifacts/profiling/` and `notebooks/`, not in core modules.

## Build, Test, and Development Commands
Prefer the Docker workflow documented in `DEV_CONTAINER_WORKFLOW.md`.

- `docker build -t nwm:cu126 .`: build the CUDA-enabled development image.
- `./scripts/docker/nwm-start.sh`: create or start the reusable `nwm_dev` container.
- `./scripts/docker/nwm-run.sh "python scripts/train.py --config configs/experiment/nwm_cdit_xl.yaml"`: run training inside the container.
- `./scripts/docker/nwm-run.sh "python scripts/infer.py ..."`: run inference or rollout generation.
- `./scripts/docker/nwm-run.sh "python scripts/evaluate.py ..."`: compute LPIPS, DreamSim, and FID metrics.
- `micromamba create -n nwm -f env.yaml`: host-side fallback when Docker is not used.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and keep imports, logging, and argument parsing consistent with existing `scripts/` entrypoints and `src/` modules. Prefer `snake_case` for functions, variables, and YAML keys; use `PascalCase` for classes such as `TrainingDataset` and `CDiTBlock`. Keep new config files and run names descriptive, for example `configs/experiment/nwm_cdit_l.yaml`. No formatter or linter is enforced in-repo, so match surrounding style and keep comments brief and technical.

## Testing Guidelines
There is no formal `pytest` suite yet. Validate changes with targeted runtime checks:

- `./scripts/docker/nwm-run.sh "python tests/smoke/recon_smoke_test.py --skip-forward"`: verify dataset loading.
- `./scripts/docker/nwm-run.sh "python tests/smoke/recon_smoke_test.py --horizon-steps 8"`: verify one-sample forward inference.
- Re-run the relevant training, inference, or evaluation command when touching model, dataset, or metric code.

Document the dataset, config, checkpoint, and GPU assumptions used for validation.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `Fix quoted Docker device GPU requests`, with occasional conventional prefixes like `fix:` and `chore:`. Keep the subject line concise, specific, and action-oriented. For pull requests, follow `CONTRIBUTING.md`: branch from `main`, add tests or smoke checks for code changes, update docs for API/workflow changes, ensure the relevant checks pass, and include a clear reproduction or evaluation summary. Link issues when applicable; attach plots or screenshots only when they clarify evaluation or notebook output changes.
