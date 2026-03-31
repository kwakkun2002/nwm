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
# Diffusion timestep respacing — 원본 diffusion process에서 timestep 부분집합을 선택하여
# 샘플링 속도를 높이는 모듈.
#
# Diffusion 모델은 학습 시 전체 T step(예: 1000)을 사용하지만, 추론 시에는 모든 step을
# 거칠 필요가 없다. 이 모듈은 원본 schedule에서 일부 timestep만 골라 새로운 축약된
# diffusion process를 구성함으로써 샘플링 횟수를 줄인다.
#
# ── 주요 구성 요소 ──
#
# 1. space_timesteps(num_timesteps, section_counts):
#    원본 T개의 timestep에서 사용할 부분집합을 결정하는 함수.
#
#    지원하는 입력 형식:
#    - 리스트 (예: [10, 15, 20]):
#      원본 timestep을 균등한 구간으로 나누고, 각 구간에서 지정된 수의 step을 균등 간격으로
#      선택한다. 예를 들어 T=300, section_counts=[10,15,20]이면:
#      · 구간 [0,99]에서 10개, [100,199]에서 15개, [200,299]에서 20개 선택 → 총 45 step
#    - 단일 숫자 문자열 (예: "250"):
#      T=1000 전체를 하나의 구간으로 보고 250개를 균등 간격으로 선택
#    - "ddimN" 형식 (예: "ddim50"):
#      DDIM 논문의 고정 stride 방식으로 정확히 N개의 step을 선택.
#      range(0, T, stride)에서 결과가 정확히 N개가 되는 정수 stride를 탐색.
#
#    반환값: 선택된 원본 timestep 인덱스의 set
#
# 2. SpacedDiffusion(GaussianDiffusion):
#    GaussianDiffusion을 상속하여 respaced timestep으로 동작하는 diffusion 클래스.
#
#    초기화 과정:
#    - 원본 betas로 GaussianDiffusion을 임시 생성하여 alphas_cumprod를 계산
#    - 선택된 timestep들에 대해 new_betas를 역산:
#      new_beta_i = 1 - ᾱ_{t_i} / ᾱ_{t_{i-1}}
#      이렇게 하면 축약된 schedule의 alphas_cumprod가 원본의 선택된 지점과 정확히 일치
#    - timestep_map: 축약된 인덱스 → 원본 인덱스 매핑 리스트
#
#    _wrap_model()을 통해 모델 호출 시 축약된 timestep을 원본 timestep으로 변환.
#    이를 통해 학습된 모델을 수정 없이 respaced sampling에 사용할 수 있다.
#
# 3. _WrappedModel:
#    축약된 timestep 인덱스를 원본 인덱스로 매핑하는 모델 래퍼.
#    SpacedDiffusion의 reverse process에서 모델이 호출될 때, 내부적으로
#    timestep_map[t]를 통해 원본 timestep을 모델에 전달한다.
#    예: 축약 schedule의 step 3 → 원본의 step 120이면, 모델에는 t=120이 전달됨.
#
# NWM 프로젝트에서의 사용:
#   __init__.py의 create_diffusion()에서 SpacedDiffusion을 생성하며, 이것이 학습과
#   추론 모두에서 사용되는 실제 diffusion 객체이다. 학습 시에는 전체 1000 step을 사용하고
#   (section_counts=[1000]), 추론 시에는 timestep_respacing 인자를 통해 step 수를
#   줄여 빠른 샘플링을 수행한다.

import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        # self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        # if self.rescale_timesteps:
        #     new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
