import os

import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from src.config import load_data_config
from src.data.transforms.action import get_action_torch
from src.data.transforms.image import unnormalize


def plot_images_with_losses(preds, losses, save_path="predictions_with_losses.png"):
    preds = (preds + 1) / 2
    ncol = int(preds.size(0) ** 0.5)
    nrow = preds.size(0) // ncol
    if ncol * nrow < preds.size(0):
        nrow += 1
    grid_img = vutils.make_grid(preds, nrow=ncol, padding=2)
    np_grid = grid_img.to(torch.float32).permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(50, 50))
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = np_grid.shape[0] // nrow, np_grid.shape[1] // ncol

    for idx, loss in enumerate(losses):
        row = idx // ncol
        col = idx % ncol
        x = col * img_width
        y = row * img_height
        if idx == 0:
            text = "GT Goal"
        else:
            text = f"ID: {idx - 1}  Loss: {loss:.2f}"
        ax.text(
            x + img_width / 2,
            y + 15,
            text,
            color="white",
            ha="center",
            va="top",
            fontsize=50,
            backgroundcolor="black",
        )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_batch_final(init_imgs, pred_imgs, goal_imgs, idxs, losses, save_path="final_plan.png"):
    imgs_for_plotting = torch.cat([init_imgs, pred_imgs, goal_imgs])
    imgs_for_plotting = (imgs_for_plotting + 1) / 2
    ncol = init_imgs.shape[0]
    grid_img = vutils.make_grid(imgs_for_plotting, nrow=ncol, padding=2)
    np_grid = grid_img.to(torch.float32).permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(ncol * 10, 30))
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = np_grid.shape[0] // 3, np_grid.shape[1] // ncol

    for i in range(ncol):
        x = i * img_width
        y_pred = img_height
        ax.text(
            x + img_width / 2,
            y_pred + 15,
            f"ID: {int(idxs[i].item())} Loss: {losses[i]:.2f}",
            color="white",
            ha="center",
            va="top",
            fontsize=40,
            backgroundcolor="black",
        )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_planning_pred(dataset_save_output_dir, idxs, obs_image, goal_image, preds, deltas, loss, gt_actions, plan_iter=0):
    for batch_idx, idx in enumerate(idxs.flatten()):
        sample_idx = int(idx)
        sample_folder = os.path.join(dataset_save_output_dir, f"id_{sample_idx}")
        os.makedirs(sample_folder, exist_ok=True)

        preds_save = {
            "obs_image": obs_image[batch_idx],
            "goal_image": goal_image[batch_idx],
            "nwm_preds": preds[batch_idx],
            "deltas": deltas[batch_idx],
            "loss": loss[batch_idx],
            "gt_actions": gt_actions[batch_idx],
        }
        preds_file = os.path.join(sample_folder, f"preds_{plan_iter}.pth")
        torch.save(preds_save, preds_file)


def log_viz_single(dataset_name, obs_image, goal_image, preds, deltas, loss, min_idx, actions, action_stats, plan_iter=0, output_dir="plot.png"):
    viz_obs_image = unnormalize(obs_image.detach().cpu())[-1]
    viz_goal_image = unnormalize(goal_image.detach().cpu())
    deltas = deltas.detach().cpu()
    loss = loss.detach().cpu()
    actions = actions.detach().cpu()
    pred_actions = get_action_torch(deltas[:, :, :2], action_stats)
    plot_array = plot_images_and_actions(dataset_name, viz_obs_image, viz_goal_image, pred_actions, actions, min_idx, loss=loss)

    plt.imshow(plot_array)
    plt.axis("off")
    plt.savefig(output_dir, format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_images_and_actions(dataset_name, curr_viz_obs_image, curr_viz_goal_image, curr_viz_pred_actions, curr_viz_actions, min_idx, loss):
    data_config = load_data_config()
    curr_viz_obs_image = curr_viz_obs_image.permute(1, 2, 0).cpu().numpy()
    curr_viz_goal_image = curr_viz_goal_image.permute(1, 2, 0).cpu().numpy()

    curr_viz_pred_actions = curr_viz_pred_actions * data_config[dataset_name]["metric_waypoint_spacing"]
    curr_viz_actions = curr_viz_actions * data_config[dataset_name]["metric_waypoint_spacing"]

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    axs[0].imshow(curr_viz_obs_image)
    axs[0].set_title("Condition Image", fontsize=13)
    axs[0].axis("off")

    axs[1].imshow(curr_viz_goal_image)
    axs[1].set_title("Goal Image", fontsize=13)
    axs[1].axis("off")

    colors = ["red", "orange", "cyan"]
    for i in range(1, curr_viz_pred_actions.shape[0]):
        color = colors[(i - 1) % len(colors)]
        label = f"Sample {i} Min Loss" if i == min_idx.item() else f"{i}"

        if i != min_idx.item():
            axs[2].plot(
                -curr_viz_pred_actions[i, :, 1],
                curr_viz_pred_actions[i, :, 0],
                color=color,
                marker="o",
                markersize=5,
                label=label,
            )
            axs[2].text(
                -curr_viz_pred_actions[i, -1, 1],
                curr_viz_pred_actions[i, -1, 0],
                round(loss[i].item(), 3),
                color="black",
                fontsize=10,
                ha="left",
                va="bottom",
            )

    axs[2].plot(
        -curr_viz_pred_actions[min_idx.item(), :, 1],
        curr_viz_pred_actions[min_idx.item(), :, 0],
        color="green",
        marker="o",
        markersize=5,
        label=f"{min_idx.item()}",
    )
    axs[2].text(
        -curr_viz_pred_actions[min_idx.item(), -1, 1],
        curr_viz_pred_actions[min_idx.item(), -1, 0],
        round(loss[min_idx.item()].item(), 3),
        color="black",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    axs[2].plot(-curr_viz_actions[:, 1], curr_viz_actions[:, 0], color="blue", marker="o", label="GT")
    axs[2].set_title("   ", fontsize=13)
    axs[2].set_xlabel("X (m)", fontsize=11)
    axs[2].set_ylabel("Y (m)", fontsize=11)
    axs[2].set_aspect("equal", adjustable="box")

    x_min, x_max = axs[2].get_xlim()
    y_min, y_max = axs[2].get_ylim()
    axis_range = max(x_max - x_min, y_max - y_min) / 2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    axs[2].set_xlim(x_mid - axis_range, x_mid + axis_range)
    axs[2].set_ylim(y_mid - axis_range, y_mid + axis_range)

    axs[2].legend(loc="lower left", fontsize=10, frameon=True, bbox_to_anchor=(0, 0))
    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    plot_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    plot_array = plot_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return plot_array
