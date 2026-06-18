#!/usr/bin/env python3
"""
RECON 데이터셋 로딩과 1-sample inference를 빠르게 확인하는 스모크 테스트.

자주 쓰는 예시:

1. 데이터셋 로딩만 확인
   python tests/smoke/recon_smoke_test.py --skip-forward

2. 기본 XL 체크포인트로 한 장 예측까지 확인
   python tests/smoke/recon_smoke_test.py --horizon-steps 8

3. 다른 샘플 / 다른 체크포인트로 확인
   python tests/smoke/recon_smoke_test.py --sample-index 3
   python tests/smoke/recon_smoke_test.py --checkpoint weights/checkpoints/nwm_cdit_s/0100000.pth.tar
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import compose_hydra_config, load_experiment_config, namespace_from_config, update_runtime_section
from src.data.datasets.factory import build_eval_dataset
from src.data.transforms.image import unnormalize
from src.diffusion import create_diffusion
from src.evaluation.inference.rollout import model_forward_wrapper
from src.models.backbones.cdit import CDiT_models
from src.models.checkpoints.vae import load_vae


def load_config(eval_config_path, model_config_path):
    return load_experiment_config(model_config_path, default_config_path=eval_config_path)


def save_tensor_image(image_tensor, output_path):
    image = unnormalize(image_tensor.detach().cpu()).clamp(0, 1)
    image = (image * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(image).save(output_path)


def write_load_report(args, dataset, idx, obs, pred, delta):
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "load_report.json")
    report = {
        "dataset": args.dataset,
        "eval_type": args.eval_type,
        "dataset_len": len(dataset),
        "sample_idx": int(idx.item()),
        "obs_shape": list(obs.shape),
        "pred_shape": list(pred.shape),
        "delta_shape": list(delta.shape),
        "obs_min": float(obs.min()),
        "obs_max": float(obs.max()),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("load_report =", report_path)


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
    vae = load_vae(device)

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


def build_parser():
    parser = argparse.ArgumentParser(
        description="RECON 데이터셋 로딩과 1-sample inference를 확인하는 스모크 테스트.",
        epilog=(
            "예시:\n"
            "  python tests/smoke/recon_smoke_test.py --skip-forward\n"
            "  python tests/smoke/recon_smoke_test.py --horizon-steps 8\n"
            "  python tests/smoke/recon_smoke_test.py --sample-index 3\n"
            "  python tests/smoke/recon_smoke_test.py --checkpoint weights/checkpoints/nwm_cdit_s/0100000.pth.tar"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--eval-config",
        default="configs/evaluation/eval_config.yaml",
        help="평가용 데이터셋 설정 파일 경로. 기본값은 RECON raw 경로가 들어간 eval 설정을 사용합니다.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/experiment/nwm_cdit_xl.yaml",
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
        default="weights/checkpoints/nwm_cdit_xl/0100000.pth.tar",
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
        default="artifacts/smoke/recon",
        help="예측 이미지를 저장할 프로젝트 내부 경로입니다.",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="모델 forward는 건너뛰고 데이터셋 로딩/shape 확인만 합니다.",
    )
    return parser


def load_smoke_config(args):
    if getattr(args, "runtime_config", None) is not None:
        return update_runtime_section(
            args.runtime_config,
            "smoke",
            args,
            (
                "eval_config",
                "model_config",
                "dataset",
                "eval_type",
                "sample_index",
                "checkpoint",
                "horizon_steps",
                "device",
                "output_dir",
                "skip_forward",
            ),
        )
    return load_config(args.eval_config, args.model_config)


def run_hydra_cli(argv):
    config = compose_hydra_config(argv)
    args = namespace_from_config(config, "smoke")
    main(args)


def uses_legacy_cli(argv):
    legacy_flags = {
        "--eval-config",
        "--model-config",
        "--dataset",
        "--eval-type",
        "--sample-index",
        "--checkpoint",
        "--horizon-steps",
        "--device",
        "--output-dir",
        "--skip-forward",
        "-h",
        "--help",
    }
    return not argv or any(arg in legacy_flags or arg.split("=", 1)[0] in legacy_flags for arg in argv)


def main(args=None):
    if args is None:
        argv = sys.argv[1:]
        if uses_legacy_cli(argv):
            args = build_parser().parse_args(argv)
        else:
            return run_hydra_cli(argv)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_smoke_config(args)
    dataset = build_eval_dataset(config, args.dataset, args.eval_type)

    sample = dataset[args.sample_index]
    if len(sample) == 5:
        idx, obs, pred, delta, _ = sample
    else:
        idx, obs, pred, delta = sample
    print("dataset_len =", len(dataset))
    print("sample_idx =", int(idx.item()))
    print("obs_shape =", tuple(obs.shape))
    print("pred_shape =", tuple(pred.shape))
    print("delta_shape =", tuple(delta.shape))
    print("obs_range =", float(obs.min()), float(obs.max()))

    if args.skip_forward:
        write_load_report(args, dataset, idx, obs, pred, delta)
        return

    run_forward(config, args, obs, delta)


if __name__ == "__main__":
    main()
