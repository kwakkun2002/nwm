# Scripts Layout

- `docker/`: container lifecycle and in-container execution helpers
- `preprocess/text/`: captioning, text cleaning, text embedding, dense alignment, and reporting
- `preprocess/recon/`: RECON-specific data preparation and rendering
- `preprocess/scand/`: SCAND download and bag conversion helpers
- `planning/`: navigation ranker dataset/training helpers
- `eval/`: checkpoint sweep and evaluation helpers
- `analysis/`: metric tables, plots, and deck/report generation
- `maintenance/`: artifact cleanup utilities
- `profiling/`: performance and GPU profiling utilities

Common entry points:

- `./scripts/docker/nwm-start.sh`
- `./scripts/docker/nwm-run.sh`
- `python scripts/train.py --config configs/experiment/nwm_cdit_s_recon_128.yaml`
- `python scripts/infer.py --exp configs/experiment/nwm_cdit_s_recon_128.yaml`
- `python scripts/evaluate.py --exp configs/experiment/nwm_cdit_s_recon_128.yaml`
- `python tests/smoke/recon_smoke_test.py --skip-forward`
- `python scripts/preprocess/text/generate_qwen_captions.py`
