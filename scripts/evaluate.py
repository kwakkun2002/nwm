# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# isolated_nwm_infer.py로 생성된 예측 이미지와 ground truth 이미지 간의
# 지각적 유사도 metric을 계산하는 독립 평가 스크립트.
#
# 여러 시간 구간(1초, 2초, 4초, 8초, 16초)에서 세 가지 metric을 계산:
#   - LPIPS (Learned Perceptual Image Patch Similarity)
#   - DreamSim (지각적 거리)
#   - FID (Frechet Inception Distance)
#
# 두 가지 평가 모드 지원:
#   - 'time': 특정 미래 시간 오프셋에서의 단일 timestep 예측 품질 평가.
#   - 'rollout': 주어진 FPS에서의 autoregressive rollout 예측 품질 평가.
#
# 결과는 데이터셋 및 평가 유형별로 JSON 파일로 저장됨.
#
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import argparse
import numpy as np

import src.core.env.distributed as dist
from src.evaluation.metrics.perceptual import get_loss_fn, evaluate, save_metric_to_disk


def main(args):
    device = 'cuda'
          
    # Loading Datasets
    dataset_names = args.datasets.split(',')
    
    secs = np.array([2**i for i in range(0, args.num_sec_eval)])
    
    # These loss functions do not accumulate
    lpips_loss_fn = get_loss_fn('lpips', secs, device)
    dreamsim_loss_fn = get_loss_fn('dreamsim', secs, device)

    for dataset_name in dataset_names:
        gt_dataset_dir = os.path.join(args.gt_dir, dataset_name)
        exp_dataset_dir = os.path.join(args.exp_dir, dataset_name)
        
        if 'rollout' in args.eval_types:
            for rollout_fps in args.rollout_fps_values:
                try:
                    metric_logger = dist.MetricLogger(delimiter="  ")
                    print("Evaluating rollout", rollout_fps, dataset_name)
                    # Rollout (LPIPS, DreamSim, FID)
                    eval_name = f'rollout_{rollout_fps}fps'
                    gt_dataset_rollout_dir = os.path.join(gt_dataset_dir, eval_name)
                    exp_dataset_rollout_dir = os.path.join(exp_dataset_dir, eval_name)
                    rollout_fid_loss_fn = get_loss_fn('fid', secs, device)
                    rollout_loss_fns = (lpips_loss_fn, dreamsim_loss_fn, rollout_fid_loss_fn)
                    with torch.no_grad():
                        evaluate(args, dataset_name, 'rollout', metric_logger, rollout_loss_fns, gt_dataset_rollout_dir, exp_dataset_rollout_dir, secs, rollout_fps)
                    output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
                    save_metric_to_disk(metric_logger, output_fn)
                except Exception as e:
                    print(e)

        if 'time' in args.eval_types:
            try:
                metric_logger = dist.MetricLogger(delimiter="  ")
                print("Evaluating time", dataset_name)
                eval_name = 'time'
                gt_dataset_time_dir = os.path.join(gt_dataset_dir, eval_name)
                exp_dataset_time_dir = os.path.join(exp_dataset_dir, eval_name)
                time_fid_loss_fn = get_loss_fn('fid', secs, device)
                time_loss_fns = (lpips_loss_fn, dreamsim_loss_fn, time_fid_loss_fn)
                with torch.no_grad():
                    evaluate(args, dataset_name, eval_name, metric_logger, time_loss_fns, gt_dataset_time_dir, exp_dataset_time_dir, secs, None)
                output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
                save_metric_to_disk(metric_logger, output_fn)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--eval_types", type=str, default='time,rollout,rollout_video', help="evluations")
    parser.add_argument("--gt_dir", type=str, default=None, help="gt directory")
    parser.add_argument("--exp_dir", type=str, default=None, help="experiment directory")
    parser.add_argument("--num_sec_eval", type=int, default=5, help="experiment name")
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    
    parser.add_argument("--input_fps", type=int, default=4, help="experiment name")
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    
    args = parser.parse_args()
    
    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]
    
    args.eval_types = args.eval_types.split(',')
    
    main(args)
