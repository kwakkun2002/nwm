# Artifact Layout

`artifacts/` is split by retention policy. Keep large reproducible outputs out of
the top level so the directory stays scannable.

## Layout

- `artifacts/bulk/train/<run_name>/<step>/`: training sample dumps.
- `artifacts/bulk/eval/<suite>/<run_name>_<checkpoint>/`: raw inference frames and GT frames.
- `artifacts/bulk/planning/<suite>/<run_name>/`: planning prediction tensors and plots.
- `artifacts/summaries/eval/...`: metric JSON, sweep manifests, and compact eval summaries.
- `artifacts/summaries/preprocess/...`: manifests, reports, and preprocessing logs.
- `artifacts/profiling/...`: long-lived profiling CSV/PNG outputs.
- `artifacts/smoke/...`: smoke-test outputs.
- `artifacts/_trash/<timestamp>/`: quarantined legacy or bulk outputs pending deletion.

## Cleanup

Preview a cleanup before moving anything:

```bash
python scripts/maintenance/clean_artifacts.py --dry-run
```

Quarantine legacy top-level outputs while preserving summaries:

```bash
python scripts/maintenance/clean_artifacts.py --execute
```

If Docker-created artifacts are owned by `root`, the cleanup script preserves
their summaries but skips moving the original directories. Fix ownership from
the host or container first, then rerun the command.

Delete quarantined files after a review period:

```bash
python scripts/maintenance/clean_artifacts.py --purge-trash --older-than-days 7 --execute
```

## Retention

Preserve metric JSON, CSV, manifests, report HTML and its local assets, and
profiling plots. Treat raw rollout frames, GT frame dumps, training sample
images, and planning `preds_*.pth` tensors as reproducible bulk artifacts.
