import numpy as np
import torch

from src.config import load_data_config, load_planning_hyperparams
from src.data.transforms.action import calculate_delta_yaw, unnormalize_data


ACTION_STATS_TORCH = {
    key: torch.tensor(value)
    for key, value in load_data_config()["action_stats"].items()
}


def initial_action_distribution(dataset_name: str, action_sampler: str, n_evals: int, traj_len: int, device=None):
    data_hyperparams = load_planning_hyperparams()
    action_mu = torch.tensor(data_hyperparams[dataset_name]["mu"], dtype=torch.float32, device=device)
    action_sigma = torch.tensor(data_hyperparams[dataset_name]["var_scale"], dtype=torch.float32, device=device)

    if action_sampler == "repeat":
        mu = action_mu.unsqueeze(0).repeat(n_evals, 1)
        sigma = action_sigma.unsqueeze(0).repeat(n_evals, 1)
        return mu, sigma

    xy_mu = action_mu[:2].repeat(traj_len)
    xy_sigma = action_sigma[:2].repeat(traj_len)
    mu = torch.cat((xy_mu, action_mu[2:3])).unsqueeze(0).repeat(n_evals, 1)
    sigma = torch.cat((xy_sigma, action_sigma[2:3])).unsqueeze(0).repeat(n_evals, 1)
    return mu, sigma


def action_params_to_deltas(action_params, len_traj_pred: int, action_sampler: str):
    if action_sampler == "repeat":
        xy_deltas = action_params[:, :2].unsqueeze(1).repeat(1, len_traj_pred, 1)
        final_yaw_offset = action_params[:, -1]
    else:
        xy_dim = len_traj_pred * 2
        xy_deltas = action_params[:, :xy_dim].reshape(-1, len_traj_pred, 2)
        final_yaw_offset = action_params[:, xy_dim]

    xy_deltas = xy_deltas.clamp(-1.0, 1.0)
    unnorm_deltas = unnormalize_data(xy_deltas, ACTION_STATS_TORCH)
    delta_yaw = calculate_delta_yaw(unnorm_deltas)
    deltas = torch.cat((xy_deltas, delta_yaw.to(xy_deltas.device)), dim=-1)
    deltas[:, -1, -1] += final_yaw_offset.clamp(-1.0, 1.0) * np.pi
    return deltas


def action_regularization_cost(deltas, smoothness_weight: float):
    cost = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    if smoothness_weight > 0:
        step_delta = deltas[:, 1:, :2] - deltas[:, :-1, :2]
        smoothness = step_delta.pow(2).mean(dim=(1, 2))
        cost = cost + smoothness_weight * smoothness
    return cost
