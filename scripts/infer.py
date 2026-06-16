# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# CDiT world model을 사용하여 예측을 생성하는 독립 추론 스크립트.
#
# 핵심 기능:
#   - model_forward_wrapper(): 전체 추론 파이프라인을 래핑 — 입력 관측의 VAE 인코딩,
#     CDiT 모델을 통한 DDPM diffusion 샘플링, VAE 디코딩으로 픽셀 공간 복원.
#     평가 및 planning 스크립트 모두에서 사용됨.
#   - generate_time(): 지수적으로 간격된 미래 시간 오프셋(1초, 2초, 4초, ...)에서
#     action delta를 누적하여 단일 timestep 예측을 생성.
#   - generate_rollout(): 다음 프레임을 예측하고 이를 context로 다시 입력하는
#     반복 방식으로 autoregressive trajectory rollout을 생성.
#
# 평가 기준선 구축을 위한 ground truth 추출 모드(--gt 1)와 분산 데이터 병렬
# 처리를 지원하는 모델 예측 모드를 모두 지원.
#
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
from pathlib import Path
import numpy as np

from src.diffusion import create_diffusion
from src.config import compose_hydra_config, int_list, load_runtime_config, namespace_from_config, save_yaml_config, str_list, update_runtime_section
from src.core.paths import DEFAULT_EVAL_ARTIFACT_ROOT, get_checkpoint_path
from src.models.checkpoints.vae import load_vae
import src.core.env.distributed as dist
from src.models.backbones.cdit import CDiT_models
from src.features.text.pipeline import get_text_conditioning_config, override_text_embedding_root
from src.data.datasets.factory import build_eval_dataset
from src.evaluation.metrics.logger import MetricLogger
from src.evaluation.inference.rollout import (
    generate_rollout, generate_time,
)

@torch.no_grad()
def main(args):
    _, _, device, _ = dist.init_distributed()
    print(args)
    device = torch.device(device)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    config = override_text_embedding_root(
        load_runtime_config(args, config_attr="exp"),
        getattr(args, "text_embedding_root", None),
    )
    if args.exp is None:
        args.exp = config["run_name"]
    args.ckp = str(args.ckp)
    args.rollout_fps_values = int_list(args.rollout_fps_values)
    dataset_names = str_list(args.datasets)
    if not dataset_names:
        raise ValueError("At least one dataset is required. Pass --datasets or infer.datasets.")
    if args.eval_type not in ("time", "rollout"):
        raise ValueError("eval_type must be either 'time' or 'rollout'.")
    config = update_runtime_section(
        config,
        "infer",
        args,
        ("output_dir", "exp", "ckp", "num_sec_eval", "input_fps", "datasets", "num_workers", "batch_size", "eval_type", "rollout_fps_values", "gt"),
    )
    exp_eval = args.exp
    if args.output_dir is None:
        args.output_dir = os.path.join(DEFAULT_EVAL_ARTIFACT_ROOT, "manual")

    # model & config setup
    if args.gt:
        args.save_output_dir = os.path.join(args.output_dir, 'gt')
    else:
        exp_name = os.path.basename(exp_eval).split('.')[0]
        args.save_output_dir = os.path.join(args.output_dir, exp_name)
    
    if  args.ckp != '0100000':
        args.save_output_dir = args.save_output_dir + "_%s"%(args.ckp)

    os.makedirs(args.save_output_dir, exist_ok=True)
    if global_rank == 0:
        save_yaml_config(config, Path(args.save_output_dir) / "resolved_config.yaml")
    text_config = get_text_conditioning_config(config)

    latent_size = config['image_size'] // 8
    args.latent_size = config['image_size'] // 8

    num_cond = config['context_size']
    print("loading")
    model_lst = (None, None, None)
    if not args.gt:
        model = CDiT_models[config['model']](
            context_size=num_cond,
            input_size=latent_size,
            in_channels=4,
            text_dim=text_config["text_dim"] if text_config["enabled"] else 0,
        )
        checkpoint_path = get_checkpoint_path(config, args.ckp)
        ckp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(model.load_state_dict(ckp["ema"], strict=True))
        model.eval()
        model.to(device)
        model = torch.compile(model)
        diffusion = create_diffusion(str(250))
        vae = load_vae(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)
        model_lst = (model, diffusion, vae)

    # Loading Datasets
    datasets = {}

    for dataset_name in dataset_names:
        dataset_val = build_eval_dataset(config, dataset_name, args.eval_type, predefined_index=True)

        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

        curr_data_loader = torch.utils.data.DataLoader(
                            dataset_val, sampler=sampler_val,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False
                        )
        datasets[dataset_name] = curr_data_loader

    print_freq = 1
    header = 'Evaluation: '
    metric_logger = MetricLogger(delimiter="  ")

    for dataset_name in dataset_names:
        dataset_save_output_dir = os.path.join(args.save_output_dir, dataset_name)
        os.makedirs(dataset_save_output_dir, exist_ok=True)
        curr_data_loader = datasets[dataset_name]
        
        for data_iter_step, batch in enumerate(metric_logger.log_every(curr_data_loader, print_freq, header)):
            if len(batch) == 5:
                idxs, obs_image, gt_image, delta, text_emb = batch
                text_emb = text_emb.to(device)
            else:
                idxs, obs_image, gt_image, delta = batch
                text_emb = None
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                obs_image = obs_image[:, -num_cond:].to(device)
                gt_image = gt_image.to(device)
                num_cond = config["context_size"]
                if args.eval_type == 'rollout':
                    for rollout_fps in args.rollout_fps_values:
                        curr_rollout_output_dir = os.path.join(dataset_save_output_dir, f'rollout_{rollout_fps}fps')
                        os.makedirs(curr_rollout_output_dir, exist_ok=True)
                        generate_rollout(args, curr_rollout_output_dir, rollout_fps, idxs, model_lst, obs_image, gt_image, delta, num_cond, device, text_emb=text_emb)
                elif args.eval_type == 'time':
                    secs = np.array([2**i for i in range(0, args.num_sec_eval)])
                    curr_time_output_dir = os.path.join(dataset_save_output_dir, 'time')
                    os.makedirs(curr_time_output_dir, exist_ok=True)
                    generate_time(args, curr_time_output_dir, idxs, model_lst, obs_image, gt_image, delta, secs, num_cond, device, text_emb=text_emb)
    

def build_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default='0100000')
    parser.add_argument("--num_sec_eval", type=int, default=5)
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--eval_type", type=str, default=None, help="type of evaluation has to be either 'time' or 'rollout'")
    parser.add_argument("--text_embedding_root", type=str, default=None, help="override text conditioning embedding root")
    # Rollout Evaluation Args
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    parser.add_argument("--gt", type=int, default=0, help="set to 1 to produce ground truth evaluation set")
    return parser


def run_hydra_cli(argv):
    config = compose_hydra_config(argv)
    args = namespace_from_config(config, "infer")
    main(args)


def uses_legacy_cli(argv):
    legacy_flags = {
        "--output_dir",
        "--exp",
        "--ckp",
        "--num_sec_eval",
        "--input_fps",
        "--datasets",
        "--num_workers",
        "--batch_size",
        "--eval_type",
        "--text_embedding_root",
        "--rollout_fps_values",
        "--gt",
        "-h",
        "--help",
    }
    return not argv or any(arg in legacy_flags or arg.split("=", 1)[0] in legacy_flags for arg in argv)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if uses_legacy_cli(argv):
        args = build_parser().parse_args()
        args.rollout_fps_values = int_list(args.rollout_fps_values)
        main(args)
    else:
        run_hydra_cli(argv)
