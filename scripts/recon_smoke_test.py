#!/usr/bin/env python3
"""
RECON 데이터셋 로딩과 1-sample inference를 빠르게 확인하는 스모크 테스트.

자주 쓰는 예시:

1. 데이터셋 로딩만 확인
   python scripts/recon_smoke_test.py --skip-forward

2. 기본 XL 체크포인트로 한 장 예측까지 확인
   python scripts/recon_smoke_test.py --horizon-steps 8

3. 다른 샘플 / 다른 체크포인트로 확인
   python scripts/recon_smoke_test.py --sample-index 3
   python scripts/recon_smoke_test.py --checkpoint logs/nwm_cdit_s/checkpoints/0100000.pth.tar
"""

import argparse
import os
import sys

import torch
import yaml
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import misc
from datasets import EvalDataset
from diffusion import create_diffusion
from models import CDiT_models
from isolated_nwm_infer import model_forward_wrapper


def load_config(eval_config_path, model_config_path):
    with open(eval_config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(model_config_path, "r") as f:
        config.update(yaml.safe_load(f))
    return config


def build_eval_dataset(config, dataset_name, eval_type):
    dataset_config = config["eval_datasets"][dataset_name]
    predefined_index = os.path.join("data_splits", dataset_name, "test", f"{eval_type}.pkl")
    return EvalDataset(
        data_folder=dataset_config["data_folder"],
        data_split_folder=dataset_config["test"],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
        max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
        len_traj_pred=config["eval_len_traj_pred"],
        traj_stride=config["traj_stride"],
        context_size=config["eval_context_size"],
        normalize=config["normalize"],
        transform=misc.transform,
        goals_per_obs=dataset_config.get("goals_per_obs", 4),
        predefined_index=predefined_index,
        traj_names="traj_names.txt",
    )


def save_tensor_image(image_tensor, output_path):
    image = misc.unnormalize(image_tensor.detach().cpu()).clamp(0, 1)
    image = (image * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(image).save(output_path)


def run_forward(config, args, obs, delta):
    device = torch.device(args.device)
    latent_size = config["image_size"] // 8
    num_cond = config["context_size"]

    obs_batch = obs.unsqueeze(0)[:, -num_cond:]
    delta_batch = delta.unsqueeze(0)
    curr_delta = delta_batch[:, :args.horizon_steps].sum(dim=1, keepdim=True)

    model = CDiT_models[config["model"]](
        context_size=num_cond,
        input_size=latent_size,
        in_channels=4,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(model.load_state_dict(checkpoint["ema"], strict=True))
    model.eval().to(device)

    diffusion = create_diffusion(str(250))
    vae = misc.load_vae(device)

    with torch.no_grad():
        output = model_forward_wrapper(
            (model, diffusion, vae),
            obs_batch,
            curr_delta,
            num_timesteps=args.horizon_steps,
            latent_size=latent_size,
            device=device,
            num_cond=num_cond,
            num_goals=1,
            progress=False,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_sample{args.sample_index}_t{args.horizon_steps}.png",
    )
    save_tensor_image(output[0], output_path)

    print("curr_delta_shape =", tuple(curr_delta.shape))
    print("pred_shape =", tuple(output.shape))
    print("pred_dtype =", output.dtype)
    print("pred_range =", float(output.min()), float(output.max()))
    print("saved =", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="RECON 데이터셋 로딩과 1-sample inference를 확인하는 스모크 테스트.",
        epilog=(
            "예시:\n"
            "  python scripts/recon_smoke_test.py --skip-forward\n"
            "  python scripts/recon_smoke_test.py --horizon-steps 8\n"
            "  python scripts/recon_smoke_test.py --sample-index 3\n"
            "  python scripts/recon_smoke_test.py --checkpoint logs/nwm_cdit_s/checkpoints/0100000.pth.tar"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--eval-config",
        default="config/eval_config.yaml",
        help="평가용 데이터셋 설정 파일 경로. 기본값은 RECON raw 경로가 들어간 eval 설정을 사용합니다.",
    )
    parser.add_argument(
        "--model-config",
        default="config/nwm_cdit_xl.yaml",
        help="모델 설정 파일 경로. 기본값은 XL 설정입니다.",
    )
    parser.add_argument(
        "--dataset",
        default="recon",
        help="불러올 eval dataset 이름. 현재 기본 사용 대상은 recon입니다.",
    )
    parser.add_argument(
        "--eval-type",
        default="time",
        choices=["time", "rollout"],
        help="어떤 predefined index를 쓸지 선택합니다. 보통 time으로 두면 됩니다.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="데이터셋에서 확인할 샘플 인덱스입니다.",
    )
    parser.add_argument(
        "--checkpoint",
        default="logs/nwm_cdit_xl/checkpoints/0100000.pth.tar",
        help="forward에 사용할 체크포인트 경로입니다. --skip-forward일 때는 사용하지 않습니다.",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=8,
        help="몇 step의 delta를 누적해서 한 장을 예측할지 정합니다. 기본값 8은 2초 ahead에 해당합니다.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="forward를 돌릴 디바이스입니다. GPU면 cuda, 디버그면 cpu로 줄 수 있습니다.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/recon_smoke",
        help="예측 이미지를 저장할 프로젝트 내부 경로입니다.",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="모델 forward는 건너뛰고 데이터셋 로딩/shape 확인만 합니다.",
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config(args.eval_config, args.model_config)
    dataset = build_eval_dataset(config, args.dataset, args.eval_type)

    idx, obs, pred, delta = dataset[args.sample_index]
    print("dataset_len =", len(dataset))
    print("sample_idx =", int(idx.item()))
    print("obs_shape =", tuple(obs.shape))
    print("pred_shape =", tuple(pred.shape))
    print("delta_shape =", tuple(delta.shape))
    print("obs_range =", float(obs.min()), float(obs.max()))

    if args.skip_forward:
        return

    run_forward(config, args, obs, delta)


if __name__ == "__main__":
    main()
