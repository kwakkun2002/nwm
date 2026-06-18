# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Goal-conditioned 시각 내비게이션을 위해 CDiT world model과 Cross Entropy
# Method (CEM)를 사용하는 planning 평가 스크립트.
#
# WM_Planning_Evaluator 클래스:
#   1. 가우시안 분포(mu, sigma)에서 후보 action 시퀀스를 샘플링.
#   2. 각 후보에 대해 world model을 통한 autoregressive rollout으로 미래 시각
#      관측을 예측.
#   3. LPIPS 지각적 손실로 goal 이미지 대비 각 rollout을 평가.
#   4. 상위 K개 trajectory를 선택하고 가우시안을 재적합 (CEM 업데이트).
#   5. 최적화 후, 최종 평균 action을 실행하고 trajectory metric으로 평가:
#      ATE (절대 궤적 오차), RPE (상대 자세 오차), goal 위치 오차, yaw 오차.
#
# trajectory metric 계산에 evo 라이브러리를 사용하며, torchrun을 통한
# 다중 GPU 분산 평가를 지원.
#
import os
import sys
from pathlib import Path

LOCAL_TORCH_CACHE = Path(__file__).resolve().parents[3] / "weights" / "cache" / "torch"
if LOCAL_TORCH_CACHE.exists():
    os.environ.setdefault("TORCH_HOME", str(LOCAL_TORCH_CACHE))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import lpips

from src.config import compose_hydra_config, load_runtime_config, namespace_from_config, save_yaml_config, str_list, update_runtime_section
from src.diffusion import create_diffusion
from src.data.datasets.factory import build_trajectory_eval_dataset
from src.evaluation.inference.rollout import model_forward_wrapper
from src.data.transforms.action import get_action_torch
from src.core.paths import DEFAULT_PLANNING_ARTIFACT_ROOT, get_checkpoint_path
from src.models.checkpoints.vae import load_vae
from src.evaluation.metrics.perceptual import save_metric_to_disk
from src.evaluation.metrics.logger import MetricLogger
from src.evaluation.planning.actions import (
    ACTION_STATS_TORCH,
    action_params_to_deltas,
    action_regularization_cost,
    initial_action_distribution,
)
from src.evaluation.planning.outputs import log_viz_single, plot_batch_final, plot_images_with_losses, save_planning_pred
from src.evaluation.planning.trajectory_metrics import actions_to_traj, eval_metrics
import src.core.env.distributed as dist
from src.models.backbones.cdit import CDiT_models
from src.features.text.pipeline import get_text_conditioning_config, override_text_embedding_root
from src.evaluation.planning.navigation_ranker import (
    DEFAULT_DINO_WEIGHTS,
    DinoFeatureExtractor,
    build_ranker_features,
    load_ranker_checkpoint,
    score_with_ranker,
)

class WM_Planning_Evaluator:
    def __init__(self, args):
        super().__init__()  
        self.args = args
        self.exp = args.exp
        _, _, device, _ = dist.init_distributed()
        self.device = torch.device(device)
        
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        
        # Setting up Config
        # self.exp_eval = f'{self.exp}_nomad_eval' # local paths etc.
        self.exp_eval = self.exp
        self.get_eval_name()

        self.config = update_runtime_section(
            override_text_embedding_root(
                load_runtime_config(self.args, config_attr="exp"),
                getattr(self.args, "text_embedding_root", None),
            ),
            "planning",
            self.args,
            (
                "exp",
                "ckp",
                "datasets",
                "output_dir",
                "save_preds",
                "num_workers",
                "batch_size",
                "num_samples",
                "rollout_stride",
                "topk",
                "opt_steps",
                "num_repeat_eval",
                "action_sampler",
                "cem_eval_chunk_size",
                "min_action_std",
                "action_smoothness_weight",
                "learned_cost_ckpt",
                "learned_cost_weight",
                "learned_cost_dino_weights",
                "learned_cost_dino_batch_size",
                "max_eval_samples",
                "eval_start_index",
                "plot",
            ),
        )
        if self.exp_eval is None:
            self.exp_eval = self.config["run_name"]
            self.exp = self.exp_eval
            self.args.exp = self.exp_eval
            self.config["planning"]["exp"] = self.exp_eval
        self.text_config = get_text_conditioning_config(self.config)

        latent_size = self.config['image_size'] // 8
        self.latent_size = self.config['image_size'] // 8
        self.num_cond = self.config['eval_context_size']
        
        # Metric JSONs are always written; predictions/plots reuse this root when enabled.
        if self.args.output_dir is None:
            self.args.output_dir = os.path.join(DEFAULT_PLANNING_ARTIFACT_ROOT, "manual")
        exp_name = os.path.basename(self.args.exp).split('.')[0]
        self.args.save_output_dir = os.path.join(self.args.output_dir, exp_name)
        os.makedirs(self.args.save_output_dir, exist_ok=True)
        if global_rank == 0:
            save_yaml_config(self.config, Path(self.args.save_output_dir) / "resolved_config.yaml")
                
        # Loading Datasets
        self.dataset_names = str_list(self.args.datasets)
        self.datasets = {}
        for dataset_name in self.dataset_names:
            dataset_val = build_trajectory_eval_dataset(self.config, dataset_name, predefined_index=True)
            if self.args.max_eval_samples is not None:
                start_index = min(self.args.eval_start_index, len(dataset_val))
                end_index = min(start_index + self.args.max_eval_samples, len(dataset_val))
                dataset_val = torch.utils.data.Subset(dataset_val, range(start_index, end_index))
            
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

            curr_data_loader = torch.utils.data.DataLoader(
                                dataset_val, sampler=sampler_val,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers,
                                pin_memory=True,
                                drop_last=False
                            )
            self.datasets[dataset_name] = curr_data_loader
        
        # Loading Model
        print("loading")
        model = CDiT_models[self.config['model']](
            context_size=self.num_cond,
            input_size=latent_size,
            text_dim=self.text_config["text_dim"] if self.text_config["enabled"] else 0,
            text_gate_mode=self.text_config["gate_mode"],
            text_gate_init=self.text_config["gate_init"],
        )

        checkpoint_path = get_checkpoint_path(self.config, args.ckp)
        ckp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckp["ema"], strict=True)
        model.eval()
        model.to(self.device)
        self.model = torch.compile(model)
        self.diffusion = create_diffusion(str(250))
        self.vae = load_vae(device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=False)
        self.model_without_ddp = self.model.module
         
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.mode = 'cem' # assume CEM for planning
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.opt_steps = self.args.opt_steps
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_sampler = self.args.action_sampler
        self.learned_cost_model = None
        self.learned_cost_checkpoint = None
        self.learned_cost_dino = None
        if self.args.learned_cost_ckpt:
            self.learned_cost_model, self.learned_cost_checkpoint = load_ranker_checkpoint(self.args.learned_cost_ckpt, self.device)
            metadata = self.learned_cost_checkpoint.get("dataset_metadata", {})
            dino_weights = self.args.learned_cost_dino_weights or metadata.get("dino_weights", DEFAULT_DINO_WEIGHTS)
            dino_arch = metadata.get("dino_arch", "vit_base")
            self.learned_cost_dino = DinoFeatureExtractor(dino_weights, dino_arch).to(self.device).eval()

    def init_mu_sigma(self, dataset_name, obs_0, traj_len):
        n_evals = obs_0.shape[0]
        return initial_action_distribution(dataset_name, self.action_sampler, n_evals, traj_len)

    def action_params_to_deltas(self, action_params, len_traj_pred):
        return action_params_to_deltas(action_params, len_traj_pred, self.action_sampler)

    def action_regularization_cost(self, deltas):
        return action_regularization_cost(deltas, self.args.action_smoothness_weight)

    def learned_navigation_cost(self, pred_images, goal_images, deltas):
        if self.learned_cost_model is None or self.args.learned_cost_weight == 0:
            return torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
        features = build_ranker_features(
            pred_images,
            goal_images,
            deltas.detach().cpu(),
            self.learned_cost_dino,
            ACTION_STATS_TORCH,
            dino_batch_size=self.args.learned_cost_dino_batch_size,
        )
        learned_cost = score_with_ranker(
            features,
            self.learned_cost_model,
            self.learned_cost_checkpoint,
            self.device,
        )
        return self.args.learned_cost_weight * learned_cost.to(device=deltas.device, dtype=deltas.dtype)

    def evaluate_candidate_losses(self, obs_images, goal_images, deltas, text_emb=None):
        def evaluate_chunk(obs_chunk, goal_chunk, delta_chunk, text_chunk):
            preds = self.autoregressive_rollout(
                obs_chunk,
                delta_chunk,
                self.args.rollout_stride,
                text_emb=text_chunk,
            )
            preds = preds[:, -1]
            loss = self.loss_fn(preds.to(self.device), goal_chunk.to(self.device)).flatten(0)
            loss = loss + self.learned_navigation_cost(preds, goal_chunk, delta_chunk).to(loss)
            return loss, preds

        chunk_size = int(self.args.cem_eval_chunk_size or 0)
        if chunk_size <= 0 or chunk_size >= deltas.shape[0]:
            return evaluate_chunk(obs_images, goal_images, deltas, text_emb)

        losses = []
        preds = []
        for start in range(0, deltas.shape[0], chunk_size):
            end = min(start + chunk_size, deltas.shape[0])
            text_chunk = None if text_emb is None else text_emb[start:end]
            chunk_loss, chunk_preds = evaluate_chunk(
                obs_images[start:end],
                goal_images[start:end],
                deltas[start:end],
                text_chunk,
            )
            losses.append(chunk_loss)
            preds.append(chunk_preds)
        return torch.cat(losses, dim=0), torch.cat(preds, dim=0)
        
    def generate_actions(self, dataset_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, len_traj_pred, text_emb=None):
        idx_string = "_".join(map(str, idxs.flatten().int().tolist())) 
        image_plot_dir = None
        if self.args.plot:
            if dataset_save_output_dir is None:
                raise ValueError("--plot requires an output directory for planning visualizations")
            image_plot_dir = os.path.join(dataset_save_output_dir, 'plots')
            os.makedirs(image_plot_dir, exist_ok=True)
        
        n_evals = obs_image.shape[0]
        mu, sigma = self.init_mu_sigma(dataset_name, obs_image, len_traj_pred)
        mu, sigma = mu.to(self.device), sigma.to(self.device)

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                traj_id = int(idxs.flatten()[traj].item())
                sample = (torch.randn(self.num_samples, mu.shape[-1]).to(self.device) * sigma[traj] + mu[traj])
                deltas = self.action_params_to_deltas(sample, len_traj_pred)

                cur_obs_image = obs_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1) 
                cur_goal_image = goal_image[traj].unsqueeze(0).repeat(self.args.num_samples, 1, 1, 1, 1).squeeze(1)
                
                # WM is stochastic, so we can repeat the evaluation of each trajectory and average to reduce variance
                if self.num_repeat_eval * self.num_samples > 120:
                    cur_losses = []
                    for r in range(self.num_repeat_eval):
                        cur_text_emb = None if text_emb is None else text_emb[traj:traj + 1].repeat(self.num_samples, 1, 1)
                        loss, preds = self.evaluate_candidate_losses(cur_obs_image, cur_goal_image, deltas, text_emb=cur_text_emb)
                        cur_losses.append(loss)

                    loss = torch.stack(cur_losses).mean(dim=0)
                    loss = loss + self.action_regularization_cost(deltas).to(loss)
                else:
                    expanded_deltas = deltas.repeat(self.num_repeat_eval, 1, 1) 
                    expanded_obs_image = cur_obs_image.repeat(self.num_repeat_eval, 1, 1, 1, 1) 
                    expanded_goal_image = cur_goal_image.repeat(self.num_repeat_eval, 1, 1, 1) 

                    expanded_text_emb = None if text_emb is None else text_emb[traj:traj + 1].repeat(self.num_repeat_eval * self.num_samples, 1, 1)
                    loss, preds = self.evaluate_candidate_losses(expanded_obs_image, expanded_goal_image, expanded_deltas, text_emb=expanded_text_emb)
                    loss = loss.view(self.num_repeat_eval, -1)
                    loss = loss.mean(dim=0)
                    loss = loss + self.action_regularization_cost(deltas).to(loss)

                    preds = preds[:self.args.num_samples]

                sorted_idx = torch.argsort(loss)
                topk_idx = sorted_idx[:self.topk]
                topk_action = sample[topk_idx]
                losses.append(loss[topk_idx[0]].item())   
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0, unbiased=False).clamp_min(self.args.min_action_std)

                if self.args.plot:
                    self.visualize_trajectories(dataset_name, gt_actions, image_plot_dir, i, traj, traj_id, deltas, cur_obs_image, cur_goal_image, preds, loss, topk_idx)                    
        
        # Final rollout 
        deltas = self.action_params_to_deltas(mu, len_traj_pred)

        preds = self.autoregressive_rollout(obs_image, deltas, self.args.rollout_stride, text_emb=text_emb)
        preds = preds[:, -1] # take the last predicted image

        loss = self.loss_fn(preds.to(self.device), goal_image.squeeze(1).to(self.device)).flatten(0)

        if self.args.save_preds:
            save_planning_pred(dataset_save_output_dir, idxs, obs_image, goal_image, preds, deltas, loss, gt_actions)
        
        if self.args.plot:
            img_name = os.path.join(image_plot_dir, f'FINAL_{idx_string}.png')
            plot_batch_final(obs_image[:, -1].to(self.device), preds, goal_image.squeeze(1).to(self.device), idxs, losses, save_path=img_name)

        pred_actions = get_action_torch(deltas[:, :, :2], ACTION_STATS_TORCH)
        pred_yaw = deltas[:, :, -1].sum(1)
        return pred_actions, pred_yaw

    def visualize_trajectories(self, dataset_name, gt_actions, image_plot_dir, i, traj, traj_id, deltas, cur_obs_image, cur_goal_image, preds, loss, topk_idx):
        img_for_plotting = torch.cat([cur_goal_image[0:1].to(self.device), preds])
        loss_for_plotting = torch.cat((torch.tensor([0]).to(self.device), loss))
        img_name = os.path.join(image_plot_dir, f'idx{traj_id}_iter{i}.png')
        plot_images_with_losses(img_for_plotting, loss_for_plotting, save_path=img_name)
        plot_name = os.path.join(image_plot_dir, f'idx{traj_id}_iter{i}_trajs.png')
        num_plot = self.args.num_samples
        log_viz_single(
                        dataset_name, 
                        cur_obs_image[0], 
                        cur_goal_image[0], 
                        preds[:num_plot], 
                        deltas[:num_plot], 
                        loss[:num_plot], 
                        topk_idx[0:1], 
                        gt_actions[traj], 
                        ACTION_STATS_TORCH, 
                        plan_iter=i, 
                        output_dir=plot_name
                    )
    
    def autoregressive_rollout(self, obs_image, deltas, rollout_stride, text_emb=None):
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        preds = []
        curr_obs = obs_image.clone().to(self.device)
        
        for i in range(deltas.shape[1]):
            curr_delta = deltas[:, i:i+1]
            all_models = self.model, self.diffusion, self.vae
            x_pred_pixels = model_forward_wrapper(
                all_models,
                curr_obs,
                curr_delta,
                self.args.rollout_stride,
                self.latent_size,
                num_cond=self.num_cond,
                device=self.device,
                text_emb=text_emb,
            )
            x_pred_pixels = x_pred_pixels.unsqueeze(1)
            
            curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1) # append current prediction
            curr_obs = curr_obs[:, 1:] # remove first observation
            preds.append(x_pred_pixels)
        
        preds = torch.cat(preds, 1)
        return preds
    
    def get_eval_name(self):
        # Get evaluation name for logging. Should overwrite for specific experiments
        sampler_name = "seq" if self.args.action_sampler == "sequence" else "repeat"
        smoothness_suffix = ""
        if self.args.action_smoothness_weight > 0:
            smoothness = f"{self.args.action_smoothness_weight:g}".replace(".", "p")
            smoothness_suffix = f"_SM{smoothness}"
        learned_suffix = ""
        if self.args.learned_cost_ckpt:
            learned_weight = f"{self.args.learned_cost_weight:g}".replace(".", "p")
            learned_suffix = f"_LC{learned_weight}"
        self.eval_name = f'CEM_{sampler_name}_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}{smoothness_suffix}{learned_suffix}'
        
    @torch.no_grad()
    def evaluate(self):
        
        for dataset_name in self.dataset_names:
            metric_logger = MetricLogger(delimiter="  ")
            header = 'Test:'
            eval_save_output_dir = None
            
            if self.args.save_preds or self.args.plot:
                dataset_save_output_dir = os.path.join(self.args.save_output_dir, dataset_name)
                os.makedirs(dataset_save_output_dir, exist_ok=True)
                eval_save_output_dir = os.path.join(dataset_save_output_dir, self.eval_name)
                os.makedirs(eval_save_output_dir, exist_ok=True)
            
            curr_data_loader = self.datasets[dataset_name]
            for batch in metric_logger.log_every(curr_data_loader, 1, header):
                if len(batch) == 6:
                    idxs, obs_image, goal_image, gt_actions, goal_pos, text_emb = batch
                    text_emb = text_emb.to(self.device)
                else:
                    idxs, obs_image, goal_image, gt_actions, goal_pos = batch
                    text_emb = None
                obs_image = obs_image[:, -self.num_cond:]
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    pred_actions, pred_yaw = self.generate_actions(
                        eval_save_output_dir,
                        dataset_name,
                        idxs,
                        obs_image,
                        goal_image,
                        gt_actions,
                        self.config["trajectory_eval_len_traj_pred"],
                        text_emb=text_emb,
                    )
                for i in range(len(obs_image)):
                    pred_traj_i = actions_to_traj(pred_actions[i, :, :2])
                    gt_traj_i = actions_to_traj(gt_actions[i, :, :2])
                    
                    ate, rpe_trans, _ = eval_metrics(gt_traj_i, pred_traj_i)

                    pred_final_pos = pred_actions[i, -1, :2].to('cpu') # (2,)
                    pred_final_yaw = pred_yaw[i].to('cpu') # 
                    goal_final_pos = goal_pos[i, 0, :2] # (2,)
                    goal_final_yaw = goal_pos[i, 0, -1] # (B,)
                    pos_diff_norm = torch.norm(pred_final_pos - goal_final_pos)
                    yaw_diff = pred_final_yaw - goal_final_yaw  # 
                    yaw_diff_norm = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()
                    
                    metric_logger.meters['{}_ate'.format(dataset_name)].update(ate, n=1)
                    metric_logger.meters['{}_rpe_trans'.format(dataset_name)].update(rpe_trans, n=1)
                    metric_logger.meters['{}_pos_diff_norm'.format(dataset_name)].update(pos_diff_norm, n=1)   
                    metric_logger.meters['{}_yaw_diff_norm'.format(dataset_name)].update(yaw_diff_norm, n=1)   
            output_fn = os.path.join(self.args.save_output_dir, f'{dataset_name}_{self.eval_name}.json')
            save_metric_to_disk(metric_logger, output_fn)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
            
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default="0100000", help="experiment name")
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir to save model predictions")
    parser.add_argument("--save_preds", action="store_true", default=False, help="whether to save prediction tensors or not")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--num_samples", type=int, default=10, help="num nomad samples to predict")
    parser.add_argument("--rollout_stride", type=int, default=1, help="rollout stride")
    parser.add_argument("--topk", type=int, default=5, help="top k samples to take mean and var for CEM")
    parser.add_argument("--opt_steps", type=int, default=15, help="num iterations for CEM")
    parser.add_argument("--num_repeat_eval", type=int, default=1, help="number of evals for one action")
    parser.add_argument("--action_sampler", type=str, default="sequence", choices=["sequence", "repeat"], help="sample per-step action sequences or legacy repeated actions")
    parser.add_argument("--cem_eval_chunk_size", type=int, default=0, help="evaluate CEM candidates in chunks to reduce peak memory; 0 keeps the original full-batch behavior")
    parser.add_argument("--min_action_std", type=float, default=1e-3, help="minimum CEM std after top-k refitting")
    parser.add_argument("--action_smoothness_weight", type=float, default=0.0, help="penalty weight for changes between consecutive sampled xy deltas")
    parser.add_argument("--learned_cost_ckpt", type=str, default=None, help="optional navigation ranker checkpoint for learned CEM cost")
    parser.add_argument("--learned_cost_weight", type=float, default=1.0, help="weight for learned navigation cost")
    parser.add_argument("--learned_cost_dino_weights", type=str, default=None, help="override DINO weights used by learned navigation cost")
    parser.add_argument("--learned_cost_dino_batch_size", type=int, default=64, help="DINO batch size for learned navigation cost")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="limit planning eval samples for smoke tests")
    parser.add_argument("--eval_start_index", type=int, default=0, help="first eval sample index when max_eval_samples is set")
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--text_embedding_root", type=str, default=None, help="override text conditioning embedding root")
    return parser


def run_hydra_cli(argv):
    config = compose_hydra_config(argv)
    args = namespace_from_config(config, "planning")
    evaluator = WM_Planning_Evaluator(args)
    evaluator.evaluate()


def uses_legacy_cli(argv):
    legacy_flags = {
        "--exp",
        "--ckp",
        "--datasets",
        "--output_dir",
        "--save_preds",
        "--num_workers",
        "--batch_size",
        "--num_samples",
        "--rollout_stride",
        "--topk",
        "--opt_steps",
        "--num_repeat_eval",
        "--action_sampler",
        "--cem_eval_chunk_size",
        "--min_action_std",
        "--action_smoothness_weight",
        "--learned_cost_ckpt",
        "--learned_cost_weight",
        "--learned_cost_dino_weights",
        "--learned_cost_dino_batch_size",
        "--max_eval_samples",
        "--eval_start_index",
        "--plot",
        "--text_embedding_root",
        "-h",
        "--help",
    }
    return not argv or any(arg in legacy_flags or arg.split("=", 1)[0] in legacy_flags for arg in argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if uses_legacy_cli(argv):
        parser = build_parser()
        args = parser.parse_args(argv)
        evaluator = WM_Planning_Evaluator(args)
        evaluator.evaluate()
    else:
        run_hydra_cli(argv)


if __name__ == "__main__":
    main()
