# Experiment Configuration and Tracking

The core training and evaluation entry points support two CLI modes:

- Legacy argparse flags, such as `--config configs/experiment/nwm_cdit_s.yaml`.
- Hydra overrides, such as `experiment=nwm_cdit_s train.epochs=10`.

Hydra uses `configs/config.yaml` as the root config and keeps
`hydra.job.chdir=false` so existing repository-relative paths continue to work.

## Training

Legacy commands still work:

```bash
./scripts/docker/nwm-run.sh "python scripts/train.py --config configs/experiment/nwm_cdit_s.yaml --epochs 300"
```

Hydra override mode:

```bash
./scripts/docker/nwm-run.sh "python scripts/train.py experiment=nwm_cdit_s train.epochs=300 train.torch_compile=false"
```

Common overrides:

```bash
experiment=nwm_cdit_b
train.epochs=50
train.ckpt_every=1000
train.eval_every=5000
batch_size=8
lr=5e-5
```

Each rank-0 training run writes the fully resolved config to:

```text
logs/<run_name>/resolved_config.yaml
```

## Inference and Metrics

Legacy inference:

```bash
./scripts/docker/nwm-run.sh "python scripts/infer.py --exp configs/experiment/nwm_cdit_s.yaml --datasets recon --eval_type time"
```

Hydra inference:

```bash
./scripts/docker/nwm-run.sh "python scripts/infer.py experiment=nwm_cdit_s infer.datasets=recon infer.eval_type=time infer.ckp=0100000"
```

Hydra metric calculation:

```bash
./scripts/docker/nwm-run.sh "python scripts/evaluate.py evaluate.gt_dir=artifacts/bulk/eval/manual/gt evaluate.exp_dir=artifacts/bulk/eval/manual/nwm_cdit_s evaluate.datasets=recon evaluate.eval_types=[time]"
```

Planning evaluation also accepts Hydra overrides:

```bash
./scripts/docker/nwm-run.sh "python scripts/plan_eval.py experiment=nwm_cdit_s planning.datasets=recon planning.ckp=0100000 planning.num_samples=120 planning.opt_steps=1"
```

Smoke and profiling helpers also support Hydra overrides:

```bash
./scripts/docker/nwm-run.sh "python tests/smoke/recon_smoke_test.py experiment=nwm_cdit_s_recon_128 smoke.skip_forward=true"
./scripts/docker/nwm-run.sh "python scripts/profiling/gpu_profile_baseline.py experiment=nwm_cdit_s profiling.repeat_runs=3 profiling.skip_flops=true"
./scripts/docker/nwm-run.sh "python scripts/preprocess/text/phase1_text_smoke.py experiment=nwm_cdit_s_recon_raw_text_dense text_smoke.device=cuda"
```

## WandB

WandB logging is disabled by default. Enable it through Hydra:

```bash
./scripts/docker/nwm-run.sh "python scripts/train.py experiment=nwm_cdit_s wandb.enabled=true wandb.project=nwm"
```

Only rank 0 initializes a WandB run. Logged values include:

- `train/loss`
- `train/steps_per_sec`
- `train/samples_per_sec`
- `eval/perceptual_loss`
- `eval/time_sec`
- evaluation sample images when `wandb.log_eval_images=true`

Checkpoint artifact logging is opt-in because model files are large:

```bash
wandb.log_checkpoints=true
```

For offline runs:

```bash
wandb.enabled=true wandb.mode=offline
```

## DVC

DVC scaffolding lives in `dvc.yaml`, `.dvcignore`, and `docs/dvc.md`. Start with
the smoke stages before tracking large datasets or checkpoints:

```bash
dvc repro smoke_recon_load
dvc repro smoke_recon_forward
```
