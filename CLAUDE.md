# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Navigation World Models (NWM) - CVPR 2025 (Oral) - Conditional Diffusion Transformer (CDiT) implementation for robotic navigation using world models. The model predicts future visual observations given current observations and actions.

## Environment Setup

### Docker (Recommended)
```bash
# Build image
docker build -t nwm:cu126 .

# Run container (with GPU, Jupyter port, volume mount)
docker run --rm -it --gpus '"device=1"' -p 8888:8888 -v "$PWD":/workspace/nwm nwm:cu126

# Inside container: start Jupyter notebook
cd /workspace/nwm
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

### Conda/Mamba
```bash
mamba create -n nwm python=3.10
mamba activate nwm
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
mamba install ffmpeg
pip3 install decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets
```

## Training

### Single GPU (Debug)
```bash
python train.py --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300 --torch-compile 0
```

### Multi-GPU with torchrun
```bash
export NUM_NODES=8
export HOST_NODE_ADDR=<HOST_ADDR>
export CURR_NODE_RANK=<NODE_RANK>

torchrun \
  --nnodes=${NUM_NODES} \
  --nproc-per-node=8 \
  --node-rank=${CURR_NODE_RANK} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${HOST_NODE_ADDR}:29500 \
  train.py --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300 --torch-compile 0
```

### Multi-GPU with submitit (SLURM)
```bash
python submitit_train_cw.py --nodes 8 --partition <partition_name> --qos <qos> --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300 --torch-compile 0
```

**Note on torch.compile**: Using `--torch-compile 1` provides ~40% speedup but may cause instabilities across PyTorch versions.

## Evaluation

Set results directory:
```bash
export RESULTS_FOLDER=/path/to/res_folder/
```

### Single Timestep Prediction

1. Generate ground truth (one-time):
```bash
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon,scand,sacson,tartan_drive \
    --batch_size 96 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1
```

2. Run prediction:
```bash
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp 0100000 \
    --datasets <dataset_name> \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```

3. Compute metrics:
```bash
python isolated_nwm_eval.py \
    --datasets <dataset_name> \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl \
    --eval_types time
```

### Trajectory Rollout Evaluation

1. Generate ground truth (one-time):
```bash
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon,scand,sacson,tartan_drive \
    --batch_size 96 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1 \
    --rollout_fps_values 1,4
```

2. Run rollout:
```bash
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp 0100000 \
    --datasets <dataset_name> \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --rollout_fps_values 1,4
```

3. Compute metrics:
```bash
python isolated_nwm_eval.py \
    --datasets <dataset_name> \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl \
    --eval_types rollout
```

### Planning Evaluation (Cross Entropy Method)

```bash
torchrun --nproc-per-node=8 planning_eval.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon \
    --rollout_stride 1 \
    --batch_size 1 \
    --num_samples 120 \
    --topk 5 \
    --num_workers 12 \
    --output_dir ${RESULTS_FOLDER} \
    --save_preds \
    --ckp 0100000 \
    --opt_steps 1 \
    --num_repeat_eval 3
```

## Architecture

### Core Model (CDiT)
- **models.py**: CDiT (Conditional Diffusion Transformer) implementation
  - CDiTBlock: Transformer block with adaptive layer norm (adaLN-Zero) and cross-attention to conditioning frames
  - CDiT: Main model with patch embedding, timestep/action embeddings, and transformer blocks
  - Model variants: CDiT-XL/2 (default, 1152 hidden, 28 layers), CDiT-L/2, CDiT-B/2, CDiT-S/2

### Diffusion Framework
- **diffusion/gaussian_diffusion.py**: DDPM-style diffusion process implementation
- **diffusion/respace.py**: Diffusion timestep respacing for faster sampling
- **diffusion/__init__.py**: Factory function `create_diffusion()` for diffusion object creation

### Data Pipeline
- **datasets.py**: PyTorch Dataset classes
  - TrainingDataset: Loads context frames + random goal frames with actions
  - EvalDataset: Loads context frames + sequential prediction frames
  - TrajectoryEvalDataset: Loads full trajectory sequences for rollout evaluation
  - Each trajectory stored as numbered JPG frames (0.jpg, 1.jpg, ...) + traj_data.pkl containing positions/yaw

### Training Loop
- **train.py**: Main training script with DDP support
  - Uses VAE from StabilityAI (sd-vae-ft-ema) to encode images to latent space
  - EMA model maintained for evaluation
  - Supports bfloat16 mixed precision training
  - Periodic evaluation using DreamSim perceptual loss

### Inference
- **isolated_nwm_infer.py**: Standalone inference for evaluation
  - `model_forward_wrapper()`: Wraps model forward pass with VAE encoding/decoding and diffusion sampling
  - Supports both single timestep and rollout evaluation modes

### Utilities
- **misc.py**: Data preprocessing utilities (coordinate transforms, normalization)
- **distributed.py**: Distributed training initialization helpers

## Configuration

Configs are in YAML format under `config/`:
- **nwm_cdit_xl.yaml**: Main training config (model size, datasets, hyperparameters)
- **eval_config.yaml**: Base evaluation config
- **data_config.yaml**: Dataset-specific parameters (waypoint spacing, action stats)

Key parameters:
- `context_size`: Number of observation frames (default: 4)
- `len_traj_pred`: Number of future frames to predict (default: 64)
- `image_size`: Input image resolution (default: 224)
- `batch_size`: Training batch size per GPU (default: 16)
- `goals_per_obs`: Number of goal frames per observation during training (default: 4)

## Data Structure

Expected data directory structure:
```
data/
├── <dataset_name>/
│   ├── <traj_name_1>/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   └── traj_data.pkl  # Contains position, yaw arrays
│   └── <traj_name_2>/
│       └── ...
```

Trajectory splits defined in `data_splits/<dataset_name>/train/traj_names.txt` and `data_splits/<dataset_name>/test/traj_names.txt`.

## Checkpoints

- Checkpoints saved to `logs/<run_name>/checkpoints/`
- Latest checkpoint: `latest.pth.tar`
- Periodic checkpoints: `<step>.pth.tar` (e.g., 0100000.pth.tar)
- Contains: model state, EMA model state, optimizer state, training step, epoch

## License

Creative Commons Attribution-NonCommercial 4.0 International
