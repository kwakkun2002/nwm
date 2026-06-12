# DVC Workflow

This repository uses DVC only as conservative scaffolding for large local data,
checkpoints, and repeatable smoke/evaluation commands. Source code, configs,
small split metadata, and the DVC scaffold files stay in Git.

Track these DVC files in Git:

- `.dvc/config`
- `.dvc/.gitignore`
- `.dvcignore`
- `dvc.yaml`
- `dvc.lock`
- any explicit `*.dvc` pointer files created for curated datasets,
  checkpoints, or published artifacts

Keep `.dvc/tmp/`, `.dvc/cache/`, and `.dvc/config.local` local-only; they are
already covered by `.dvc/.gitignore`.

## Setup

Install DVC on the host, then run Python through the existing Docker wrapper:

```bash
./scripts/docker/nwm-start.sh
```

The repository is already initialized with `.dvc/`. The stages in `dvc.yaml`
call `./scripts/docker/nwm-run.sh`, so `dvc repro` should be run from the host
repository checkout, not from inside the container.

## Smoke Stages

Use these for quick RECON validation:

```bash
dvc repro smoke_recon_load
dvc repro smoke_recon_forward
```

The load stage checks dataset construction and writes
`artifacts/dvc/smoke/recon_load/load_report.json`. The forward stage writes a
single prediction image to `artifacts/dvc/smoke/recon` and expects:

- `datasets/recon_raw/recon_release`
- `data/splits/recon/test`
- `weights/checkpoints/nwm_cdit_s_recon_128/latest.pth.tar`

## Evaluation Runs

Full time/rollout evaluation can be run with the Hydra-enabled scripts and then
tracked selectively with `dvc add` when the outputs are worth preserving. These
runs are intentionally not part of the default `dvc.yaml` because they can
generate large image trees and take much longer than smoke validation.

```bash
./scripts/docker/nwm-run.sh "python scripts/infer.py experiment=nwm_cdit_s_recon_128 infer.output_dir=artifacts/dvc/eval/recon_128 infer.datasets=recon infer.eval_type=time infer.num_sec_eval=3 infer.gt=true"
./scripts/docker/nwm-run.sh "python scripts/infer.py experiment=nwm_cdit_s_recon_128 infer.output_dir=artifacts/dvc/eval/recon_128 infer.datasets=recon infer.eval_type=time infer.num_sec_eval=3 infer.ckp=latest"
./scripts/docker/nwm-run.sh "python scripts/evaluate.py evaluate.gt_dir=artifacts/dvc/eval/recon_128/gt evaluate.exp_dir=artifacts/dvc/eval/recon_128/nwm_cdit_s_recon_128_latest evaluate.datasets=recon evaluate.eval_types=[time] evaluate.num_sec_eval=3"
```

After confirming the results:

```bash
dvc add artifacts/dvc/eval/recon_128/nwm_cdit_s_recon_128_latest/recon_time.json
```

## What To Track With DVC

Track stable, shareable inputs:

- Raw or preprocessed datasets under `datasets/`, such as
  `datasets/recon_raw/recon_release`.
- Large checkpoint directories or selected checkpoint files under
  `weights/checkpoints/`.
- Pretrained model and metric weights under `weights/pretrained/` when they are
  not reliably downloaded by the runtime.
- Curated, published evaluation artifacts only when the team wants to preserve
  the exact files behind a report.

Keep these out of DVC unless there is a specific reason:

- `logs/`, TensorBoard output, and training scratch directories.
- Bulk generated images and videos under `artifacts/bulk/`.
- Profiling raw dumps, PID files, temporary logs, and notebook caches.
- One-off manual debug media.

Example adds:

```bash
dvc add datasets/recon_raw/recon_release
dvc add weights/checkpoints/nwm_cdit_s_recon_128/latest.pth.tar
dvc add weights/pretrained
git add dvc.yaml .dvcignore .gitignore docs/dvc.md .dvc \
  datasets/**/*.dvc weights/**/*.dvc
```

Use `dvc push` only after confirming the remote and storage policy for the team.
