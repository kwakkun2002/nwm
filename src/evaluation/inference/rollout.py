import torch
import os
import numpy as np
from PIL import Image

from src.data.transforms.image import unnormalize, build_transform
from src.data.datasets.eval_dataset import EvalDataset
from src.features.text.pipeline import get_text_conditioning_config


def save_image(output_file, img, unnormalize_img):
    img = img.detach().cpu()
    if unnormalize_img:
        img = unnormalize(img)

    img = img * 255
    img = img.byte()
    image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')
    image.save(output_file)


def get_dataset_eval(config, dataset_name, eval_type, predefined_index=True):
    data_config = config["eval_datasets"][dataset_name]
    text_config = get_text_conditioning_config(config)
    image_transform = build_transform(config["image_size"])
    if predefined_index:
        predefined_index = f"data/splits/{dataset_name}/test/{eval_type}.pkl"
    else:
        predefined_index = None

    dataset = EvalDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
                max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
                len_traj_pred=config["eval_len_traj_pred"],
                traj_stride=config["traj_stride"],
                context_size=config["eval_context_size"],
                normalize=config["normalize"],
                transform=image_transform,
                goals_per_obs=4,
                predefined_index=predefined_index,
                traj_names='traj_names.txt',
                text_embedding_root=text_config["embedding_root"] if text_config["enabled"] else None,
                text_condition_source=text_config["condition_source"],
            )

    return dataset


@torch.no_grad()
def model_forward_wrapper(
    all_models,
    curr_obs,
    curr_delta,
    num_timesteps,
    latent_size,
    device,
    num_cond,
    num_goals=1,
    rel_t=None,
    text_emb=None,
    progress=False,
):
    model, diffusion, vae = all_models
    x = curr_obs.to(device)
    y = curr_delta.to(device)

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]

        if rel_t is None:
            rel_t = (torch.ones(B)* (1. / 128.)).to(device)
            rel_t *= num_timesteps

        x = x.flatten(0,1)
        x = vae.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
        x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
        z = torch.randn(B*num_goals, 4, latent_size, latent_size, device=device)
        y = y.flatten(0, 1)
        model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
        if text_emb is not None:
            model_kwargs["text_emb"] = text_emb.flatten(0, 1).to(device)
        samples = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=progress, device=device
        )
        samples = vae.decode(samples / 0.18215).sample

        return torch.clip(samples, -1., 1.)


def generate_rollout(args, output_dir, rollout_fps, idxs, all_models, obs_image, gt_image, delta, num_cond, device, text_emb=None):
    rollout_stride = args.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride-1::rollout_stride]
    delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)
    curr_obs = obs_image.clone().to(device)

    for i in range(gt_image.shape[1]):
        curr_delta = delta[:, i:i+1].to(device)
        if args.gt:
            x_pred_pixels = gt_image[:, i].clone().to(device)
        else:
            x_pred_pixels = model_forward_wrapper(
                all_models,
                curr_obs,
                curr_delta,
                rollout_stride,
                args.latent_size,
                num_cond=num_cond,
                num_goals=1,
                device=device,
                text_emb=text_emb,
            )

        curr_obs = torch.cat((curr_obs, x_pred_pixels.unsqueeze(1)), dim=1) # append current prediction
        curr_obs = curr_obs[:, 1:] # remove first observation
        visualize_preds(output_dir, idxs, i, x_pred_pixels)


def generate_time(args, output_dir, idxs, all_models, obs_image, gt_output, delta, secs, num_cond, device, text_emb=None):
    eval_timesteps = [sec*args.input_fps for sec in secs]
    for sec, timestep in zip(secs, eval_timesteps):
        curr_delta = delta[:, :timestep].sum(dim=1, keepdim=True)
        if args.gt:
            x_pred_pixels = gt_output[:, timestep-1].clone().to(device)
        else:
            x_pred_pixels = model_forward_wrapper(
                all_models,
                obs_image,
                curr_delta,
                timestep,
                args.latent_size,
                num_cond=num_cond,
                num_goals=1,
                device=device,
                text_emb=text_emb,
            )
        visualize_preds(output_dir, idxs, sec, x_pred_pixels)


def visualize_preds(output_dir, idxs, sec, x_pred_pixels):
    for batch_idx, sample_idx in enumerate(idxs.reshape(-1)):
        sample_idx = int(sample_idx.item())
        sample_folder = os.path.join(output_dir, f'id_{sample_idx}')
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f'{sec}.png')
        save_image(image_file, x_pred_pixels[batch_idx], True)
