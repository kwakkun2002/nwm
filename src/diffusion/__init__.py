# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# --------------------------------------------------------
# diffusion 패키지의 진입점이자 팩토리 모듈.
#
# create_diffusion() 함수를 통해 diffusion 객체를 생성하는 단일 인터페이스를 제공한다.
# 이 함수는 내부적으로 다음과 같은 설정을 조합하여 SpacedDiffusion 인스턴스를 구성한다:
#
#   1. Noise schedule 선택:
#      - "linear": Ho et al. (DDPM, 2020)의 선형 beta schedule
#      - "squaredcos_cap_v2": cosine schedule (Improved DDPM)
#      get_named_beta_schedule()을 호출하여 각 diffusion timestep의 beta 값 배열 생성.
#
#   2. Loss type 결정:
#      - MSE: epsilon 또는 x_0 예측에 대한 평균 제곱 오차 (기본값)
#      - RESCALED_MSE: learned variance 사용 시 VLB 항을 rescale하여 MSE와 결합
#      - KL / RESCALED_KL: variational lower bound 기반 손실
#
#   3. Model 출력 타입 설정:
#      - predict_xstart=False (기본): 모델이 noise epsilon을 예측 (ModelMeanType.EPSILON)
#      - predict_xstart=True: 모델이 깨끗한 이미지 x_0를 직접 예측 (ModelMeanType.START_X)
#
#   4. Variance 처리 방식:
#      - learn_sigma=True (기본): 모델이 variance 범위를 학습 (ModelVarType.LEARNED_RANGE)
#        → 모델 출력 채널이 2배가 되며, [FIXED_SMALL, FIXED_LARGE] 사이를 보간
#      - learn_sigma=False: 고정 variance 사용 (FIXED_LARGE 또는 FIXED_SMALL)
#
#   5. Timestep respacing:
#      - space_timesteps()로 원본 diffusion_steps(기본 1000)에서 사용할 timestep 부분집합 선택
#      - 예: "250" → 1000 step 중 250개를 균등 간격으로 선택하여 sampling 가속
#      - "ddim50" → DDIM 논문 방식으로 50 step 선택
#      - None 또는 "" → 모든 timestep 사용 (원본 schedule 그대로)
#
# NWM 프로젝트에서의 사용:
#   train.py에서 학습용 diffusion 객체 생성, isolated_nwm_infer.py에서 추론용 객체 생성 시
#   이 함수를 호출한다. 학습 시에는 전체 1000 step을 사용하고, 추론 시에는 respacing을 통해
#   샘플링 step 수를 줄여 속도를 높인다.

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )
