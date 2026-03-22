#!/usr/bin/env python3
import argparse
import os
import sys

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets import TrainingDataset
from diffusion import create_diffusion
from misc import load_vae, transform
from models import CDiT_models
from text_pipeline import infer_text_embedding_dim


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 cached text conditioning smoke test.")
    parser.add_argument("--config", default="config/nwm_cdit_s.yaml")
    parser.add_argument("--dataset-name", default="recon")
    parser.add_argument("--data-folder", default=None)
    parser.add_argument("--data-split-folder", default=None)
    parser.add_argument("--embedding-root", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--goals-per-obs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--len-traj-pred", type=int, default=None)
    parser.add_argument("--context-size", type=int, default=None)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config(args.config)
    distance_cfg = config.get("distance", {})
    dataset_cfg = config["datasets"][args.dataset_name]
    data_folder = args.data_folder if args.data_folder is not None else dataset_cfg["data_folder"]
    data_split_folder = args.data_split_folder if args.data_split_folder is not None else dataset_cfg["train"]
    embedding_root = args.embedding_root
    if embedding_root is None:
        embedding_root = config.get("text_conditioning", {}).get("embedding_root")
    if not embedding_root:
        raise ValueError("An embedding root is required. Pass --embedding-root or set text_conditioning.embedding_root in config.")
    text_dim = infer_text_embedding_dim(embedding_root)

    context_size = args.context_size if args.context_size is not None else config["context_size"]
    len_traj_pred = args.len_traj_pred if args.len_traj_pred is not None else config["len_traj_pred"]
    goals_per_obs = args.goals_per_obs if args.goals_per_obs is not None else int(dataset_cfg.get("goals_per_obs", 4))

    dataset = TrainingDataset(
        data_folder=data_folder,
        data_split_folder=data_split_folder,
        dataset_name=args.dataset_name,
        image_size=config["image_size"],
        min_dist_cat=distance_cfg["min_dist_cat"],
        max_dist_cat=distance_cfg["max_dist_cat"],
        len_traj_pred=len_traj_pred,
        traj_stride=1,
        context_size=context_size,
        transform=transform,
        normalize=config["normalize"],
        goals_per_obs=goals_per_obs,
        text_embedding_root=embedding_root,
        text_condition_source="current",
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset index is empty for context_size={context_size}, len_traj_pred={len_traj_pred}. "
            "Short 1fps trajectories require a shorter prediction horizon."
        )

    sample_meta = dataset.index_to_data[args.sample_index]
    sample = dataset[args.sample_index]
    if len(sample) != 4:
        raise RuntimeError(f"Expected text-conditioned training sample with 4 tensors, got {len(sample)}")

    x, y, rel_t, text_emb = sample
    print("dataset_len =", len(dataset))
    print("sample_meta =", sample_meta)
    print("obs_shape =", tuple(x.shape))
    print("goal_shape =", tuple(y.shape))
    print("rel_t_shape =", tuple(rel_t.shape))
    print("text_shape =", tuple(text_emb.shape))
    print("text_dtype =", text_emb.dtype)
    print("text_abs_sum =", float(text_emb.abs().sum()))
    print("text_dim =", text_dim)

    device = torch.device(args.device)
    model = CDiT_models[config["model"]](
        context_size=context_size,
        input_size=config["image_size"] // 8,
        in_channels=4,
        text_dim=text_dim,
    ).to(device)
    model.eval()

    vae = load_vae(device)
    diffusion = create_diffusion(timestep_respacing="")

    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    rel_t = rel_t.unsqueeze(0).to(device)
    text_emb = text_emb.unsqueeze(0).to(device)

    with torch.no_grad():
        bsz, num_frames = x.shape[:2]
        latents = vae.encode(x.flatten(0, 1)).latent_dist.sample().mul_(0.18215)
        latents = latents.unflatten(0, (bsz, num_frames))

        num_cond = context_size
        num_goals = num_frames - num_cond
        x_start = latents[:, num_cond:].flatten(0, 1)
        x_cond = latents[:, :num_cond].unsqueeze(1).expand(
            bsz, num_goals, num_cond, latents.shape[2], latents.shape[3], latents.shape[4]
        ).flatten(0, 1)
        y = y.flatten(0, 1)
        rel_t = rel_t.flatten(0, 1)
        text_emb = text_emb.flatten(0, 1)
        t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)
        loss_dict = diffusion.training_losses(
            model,
            x_start,
            t,
            {"y": y, "x_cond": x_cond, "rel_t": rel_t, "text_emb": text_emb},
        )
        loss = float(loss_dict["loss"].mean().item())

    print("latent_shape =", tuple(latents.shape))
    print("x_start_shape =", tuple(x_start.shape))
    print("x_cond_shape =", tuple(x_cond.shape))
    print("loss =", loss)
    print("smoke_status = ok")


if __name__ == "__main__":
    main()
