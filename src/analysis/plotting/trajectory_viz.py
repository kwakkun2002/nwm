import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from src.data.transforms.action import get_action_torch
from src.data.transforms.image import unnormalize

with open("configs/data/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)


def log_viz_single(dataset_name, obs_image, goal_image, preds, deltas, loss, min_idx, actions, action_stats, plan_iter=0, output_dir='plot.png'):
    '''
    Visualize a single instance
    actions is gt actions
    '''
    viz_obs_image = unnormalize(obs_image.detach().cpu())[-1] # take last img
    viz_goal_image = unnormalize(goal_image.detach().cpu())
    deltas = deltas.detach().cpu()
    loss = loss.detach().cpu()
    actions = actions.detach().cpu()
    pred_actions = get_action_torch(deltas[:, :, :2], action_stats)
    plot_array = plot_images_and_actions(dataset_name, viz_obs_image, viz_goal_image, pred_actions, actions, min_idx, loss=loss)

    plt.imshow(plot_array)
    plt.axis('off')  # Hide axes for a cleaner image

    # Save the plot array as a PNG file locally
    plt.savefig(output_dir, format='png', dpi=300, bbox_inches='tight')

def plot_images_and_actions(dataset_name, curr_viz_obs_image, curr_viz_goal_image, curr_viz_pred_actions, curr_viz_actions, min_idx, loss):
    curr_viz_obs_image = curr_viz_obs_image.permute(1, 2, 0).cpu().numpy()
    curr_viz_goal_image = curr_viz_goal_image.permute(1, 2, 0).cpu().numpy()

    # scale back to metric space for plotting
    curr_viz_pred_actions = curr_viz_pred_actions * data_config[dataset_name]['metric_waypoint_spacing']
    curr_viz_actions = curr_viz_actions * data_config[dataset_name]['metric_waypoint_spacing']

    # Create the figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    # Plot condition image
    axs[0].imshow(curr_viz_obs_image)
    axs[0].set_title("Condition Image", fontsize=13)
    axs[0].axis("off")

    # Plot goal image
    axs[1].imshow(curr_viz_goal_image)
    axs[1].set_title("Goal Image", fontsize=13)
    axs[1].axis("off")

    colors = ['red', 'orange', 'cyan']
    for i in range(1, curr_viz_pred_actions.shape[0]):
        color = colors[(i - 1) % len(colors)]
        label = f"Sample {i} Min Loss" if i == min_idx.item() else f"{i}"

        if i != min_idx.item():
            axs[2].plot(-curr_viz_pred_actions[i, :, 1], curr_viz_pred_actions[i, :, 0],
                        color=color, marker="o", markersize=5, label=label)
            axs[2].text(-curr_viz_pred_actions[i, -1, 1],
                curr_viz_pred_actions[i, -1, 0],
                round(loss[i].item(), 3),
                color='black',
                fontsize=10,
                ha='left', va='bottom')  # Adjust position to avoid overlap

    # Highlight the minimum loss sample
    axs[2].plot(-curr_viz_pred_actions[min_idx.item(), :, 1], curr_viz_pred_actions[min_idx.item(), :, 0],
                color='green', marker="o", markersize=5, label=f"{min_idx.item()}")
    axs[2].text(-curr_viz_pred_actions[min_idx.item(), -1, 1],
        curr_viz_pred_actions[min_idx.item(), -1, 0],
        round(loss[min_idx.item()].item(), 3),
        color='black',
        fontsize=10,
        ha='left', va='bottom')  # Adjust position to avoid overlap

    # Plot ground truth actions
    axs[2].plot(-curr_viz_actions[:, 1], curr_viz_actions[:, 0], color='blue', marker="o", label="GT")

    # Set titles and labels with larger font size
    axs[2].set_title("   ", fontsize=13)
    axs[2].set_xlabel("X (m)", fontsize=11)
    axs[2].set_ylabel("Y (m)", fontsize=11)

    # Set equal aspect ratio and adjust axis limits
    axs[2].set_aspect('equal', adjustable='box')
    x_min, x_max = axs[2].get_xlim()
    y_min, y_max = axs[2].get_ylim()
    axis_range = max(x_max - x_min, y_max - y_min) / 2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    axs[2].set_xlim(x_mid - axis_range, x_mid + axis_range)
    axs[2].set_ylim(y_mid - axis_range, y_mid + axis_range)

    axs[2].legend(loc='lower left', fontsize=10, frameon=True, bbox_to_anchor=(0, 0))
    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    plot_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plot_array = plot_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return plot_array
