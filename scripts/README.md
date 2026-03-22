# Scripts Layout

- `docker/`: container lifecycle and in-container execution helpers
- `text/`: captioning, text cleaning, text embedding, dense alignment, and reporting
- `recon/`: RECON-specific data preparation, rendering, and smoke tests
- `profiling/`: performance and GPU profiling utilities

Common entry points:

- `./scripts/docker/nwm-start.sh`
- `./scripts/docker/nwm-run.sh`
- `python scripts/recon/recon_smoke_test.py`
- `python scripts/text/generate_qwen_captions.py`
