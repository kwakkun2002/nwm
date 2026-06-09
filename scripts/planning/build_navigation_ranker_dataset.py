import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.transforms.action import get_action_torch, get_delta_np, normalize_data
from src.evaluation.planning.actions import ACTION_STATS_TORCH
from src.evaluation.planning.cem_planner import WM_Planning_Evaluator, build_parser
from src.evaluation.planning.navigation_ranker import (
    DEFAULT_DINO_WEIGHTS,
    DinoFeatureExtractor,
    build_ranker_features,
)


def compute_labels(deltas, gt_actions):
    pred_actions = get_action_torch(deltas[:, :, :2], ACTION_STATS_TORCH).detach().cpu().float()
    gt_actions = gt_actions.detach().cpu().float()
    gt_xy = gt_actions[:, :2].unsqueeze(0).expand_as(pred_actions[:, :, :2])

    xy_error = pred_actions[:, :, :2] - gt_xy
    ate = torch.sqrt(xy_error.pow(2).sum(dim=-1).mean(dim=1))
    final_pos_error = torch.norm(pred_actions[:, -1, :2] - gt_actions[-1, :2], dim=-1)

    pred_steps = torch.diff(torch.cat((torch.zeros(pred_actions.shape[0], 1, 2), pred_actions[:, :, :2]), dim=1), dim=1)
    gt_steps = torch.diff(torch.cat((torch.zeros(1, 2), gt_actions[:, :2]), dim=0), dim=0)
    rpe_trans = torch.sqrt((pred_steps - gt_steps.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=1))

    pred_yaw = deltas[:, :, 2].detach().cpu().float().sum(dim=1)
    gt_yaw = gt_actions[-1, 2].float()
    yaw_diff = pred_yaw - gt_yaw
    yaw_error = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()
    return {
        "ate": ate,
        "rpe_trans": rpe_trans,
        "final_pos_error": final_pos_error,
        "yaw_error": yaw_error,
    }


def labels_to_target(labels, args):
    return (
        args.label_ate_weight * labels["ate"]
        + args.label_rpe_weight * labels["rpe_trans"]
        + args.label_pos_weight * labels["final_pos_error"]
        + args.label_yaw_weight * labels["yaw_error"]
    )


def gt_deltas_from_actions(gt_actions):
    xy_delta = torch.as_tensor(get_delta_np(gt_actions[:, :2].detach().cpu().numpy()), dtype=torch.float32)
    xy_delta = normalize_data(xy_delta, {k: v.detach().cpu() for k, v in ACTION_STATS_TORCH.items()})
    yaw = gt_actions[:, 2].detach().cpu().float()
    yaw_delta = torch.diff(torch.cat((torch.zeros(1), yaw), dim=0), dim=0).unsqueeze(1)
    return torch.cat((xy_delta, yaw_delta), dim=1)


def sample_candidate_deltas(evaluator, obs_image, gt_actions, len_traj_pred, num_candidates, include_gt):
    mu, sigma = evaluator.init_mu_sigma(obs_image, len_traj_pred)
    mu = mu.to(evaluator.device)
    sigma = sigma.to(evaluator.device)
    params = torch.randn(num_candidates, mu.shape[-1], device=evaluator.device) * sigma[0] + mu[0]
    deltas = evaluator.action_params_to_deltas(params, len_traj_pred)
    if include_gt:
        gt_deltas = gt_deltas_from_actions(gt_actions).to(evaluator.device).unsqueeze(0)
        deltas = torch.cat((gt_deltas, deltas), dim=0)
    return deltas


@torch.no_grad()
def build_dataset(args):
    evaluator = WM_Planning_Evaluator(args)
    dino = DinoFeatureExtractor(args.dino_weights, args.dino_arch, args.dino_patch_size, args.dino_image_size)
    dino.to(evaluator.device).eval()

    output_dir = Path(args.ranker_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset_name}.pt"

    all_features = []
    all_targets = []
    all_groups = []
    all_candidate_index = []
    label_store = {key: [] for key in ["ate", "rpe_trans", "final_pos_error", "yaw_error"]}

    dataset_loader = evaluator.datasets[args.dataset_name]
    group_id = 0
    for batch in dataset_loader:
        if len(batch) == 6:
            idxs, obs_image, goal_image, gt_actions, goal_pos, text_emb = batch
            text_emb = text_emb.to(evaluator.device)
        else:
            idxs, obs_image, goal_image, gt_actions, goal_pos = batch
            text_emb = None
        obs_image = obs_image[:, -evaluator.num_cond:]

        for batch_idx in range(obs_image.shape[0]):
            cur_obs = obs_image[batch_idx:batch_idx + 1].to(evaluator.device)
            cur_goal = goal_image[batch_idx:batch_idx + 1].to(evaluator.device)
            cur_gt = gt_actions[batch_idx].cpu()
            cur_text_emb = None if text_emb is None else text_emb[batch_idx:batch_idx + 1]
            deltas = sample_candidate_deltas(
                evaluator,
                cur_obs,
                cur_gt,
                evaluator.config["trajectory_eval_len_traj_pred"],
                args.num_ranker_candidates,
                args.include_gt_candidate,
            )
            rollout_obs = cur_obs.repeat(deltas.shape[0], 1, 1, 1, 1)
            rollout_text = None if cur_text_emb is None else cur_text_emb.repeat(deltas.shape[0], 1, 1)
            preds = evaluator.autoregressive_rollout(
                rollout_obs,
                deltas,
                evaluator.args.rollout_stride,
                text_emb=rollout_text,
            )[:, -1]
            goals = cur_goal.repeat(deltas.shape[0], 1, 1, 1, 1).squeeze(1)

            features = build_ranker_features(
                preds,
                goals,
                deltas.detach().cpu(),
                dino,
                ACTION_STATS_TORCH,
                dino_batch_size=args.dino_batch_size,
            )
            labels = compute_labels(deltas, cur_gt)
            targets = labels_to_target(labels, args)

            all_features.append(features)
            all_targets.append(targets)
            all_groups.append(torch.full((features.shape[0],), group_id, dtype=torch.long))
            all_candidate_index.append(torch.arange(features.shape[0], dtype=torch.long))
            for key, value in labels.items():
                label_store[key].append(value)

            group_id += 1
            if group_id % args.log_every == 0:
                print(f"built {group_id} samples, candidates={sum(x.numel() for x in all_targets)}")

    payload = {
        "features": torch.cat(all_features, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "groups": torch.cat(all_groups, dim=0),
        "candidate_index": torch.cat(all_candidate_index, dim=0),
        "labels": {key: torch.cat(value, dim=0) for key, value in label_store.items()},
        "metadata": {
            "dataset": args.dataset_name,
            "exp": args.exp,
            "ckp": args.ckp,
            "num_ranker_candidates": args.num_ranker_candidates,
            "include_gt_candidate": args.include_gt_candidate,
            "action_sampler": args.action_sampler,
            "dino_weights": args.dino_weights,
            "dino_arch": args.dino_arch,
            "target_weights": {
                "ate": args.label_ate_weight,
                "rpe": args.label_rpe_weight,
                "pos": args.label_pos_weight,
                "yaw": args.label_yaw_weight,
            },
        },
    }
    torch.save(payload, output_path)

    summary = {
        "output": str(output_path),
        "num_groups": int(payload["groups"].max().item() + 1),
        "num_candidates": int(payload["features"].shape[0]),
        "feature_dim": int(payload["features"].shape[1]),
        "target_mean": float(payload["targets"].mean()),
        "target_std": float(payload["targets"].std(unbiased=False)),
    }
    with (output_path.with_suffix(".json")).open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = build_parser()
    parser.add_argument("--ranker_output_dir", type=str, default="artifacts/bulk/planning_ranker/datasets")
    parser.add_argument("--dataset_name", type=str, default="recon")
    parser.add_argument("--num_ranker_candidates", type=int, default=32)
    parser.add_argument("--include_gt_candidate", action="store_true", default=True)
    parser.add_argument("--no_include_gt_candidate", dest="include_gt_candidate", action="store_false")
    parser.add_argument("--dino_weights", type=str, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument("--dino_arch", type=str, default="vit_base", choices=["vit_base", "vit_small"])
    parser.add_argument("--dino_patch_size", type=int, default=16)
    parser.add_argument("--dino_image_size", type=int, default=224)
    parser.add_argument("--dino_batch_size", type=int, default=64)
    parser.add_argument("--label_ate_weight", type=float, default=1.0)
    parser.add_argument("--label_rpe_weight", type=float, default=0.5)
    parser.add_argument("--label_pos_weight", type=float, default=1.0)
    parser.add_argument("--label_yaw_weight", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    args.datasets = args.dataset_name
    build_dataset(args)


if __name__ == "__main__":
    main()
