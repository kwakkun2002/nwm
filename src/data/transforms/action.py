import torch
import numpy as np


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'].to(ndata) - stats['min'].to(ndata)) + stats['min'].to(ndata)
    return data

def get_action_torch(diffusion_output, action_stats):
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = torch.cumsum(ndeltas, dim=1)
    return actions.to(ndeltas)

def get_delta_np(actions):
    # append zeros to first action (unbatched)
    ex_actions = np.concatenate((np.zeros((1, actions.shape[1])), actions), axis=0)
    delta = ex_actions[1:] - ex_actions[:-1]
    return delta

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def angle_difference(theta1, theta2):
    delta_theta = theta2 - theta1
    delta_theta = delta_theta - 2 * np.pi * np.floor((delta_theta + np.pi) / (2 * np.pi))
    return delta_theta

def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def calculate_delta_yaw(unnorm_actions):
    x = unnorm_actions[..., 0]
    y = unnorm_actions[..., 1]

    yaw = torch.atan2(y, x).unsqueeze(-1)
    delta_yaw = torch.cat((torch.zeros(yaw.shape[0], 1, yaw.shape[2]).to(yaw.device), yaw), dim=1)
    delta_yaw = delta_yaw[:, 1:, :] - delta_yaw[:, :-1, :]

    return delta_yaw
