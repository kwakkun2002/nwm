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
# Diffusion 모델의 확률 분포 계산에 필요한 수학적 유틸리티 함수 모음.
#
# GaussianDiffusion 클래스에서 VLB(Variational Lower Bound) 계산 및
# 손실 함수 평가 시 호출되는 저수준 확률 연산들을 제공한다.
#
# ── 주요 함수 ──
#
# 1. normal_kl(mean1, logvar1, mean2, logvar2):
#    두 Gaussian 분포 사이의 KL divergence를 계산.
#    KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = 0.5 * (-1 + log(σ₂²/σ₁²) + σ₁²/σ₂² + (μ₁-μ₂)²/σ₂²)
#
#    사용처: _vb_terms_bpd()에서 실제 posterior q(x_{t-1}|x_t,x_0)와
#    모델이 예측한 p(x_{t-1}|x_t) 사이의 KL divergence를 계산할 때 호출.
#    스칼라와 텐서 간 broadcasting을 지원하여 prior KL 계산(mean2=0, logvar2=0)도 가능.
#
# 2. approx_standard_normal_cdf(x):
#    표준 정규분포의 누적분포함수(CDF) Φ(x)의 빠른 근사.
#    tanh 기반 근사식 사용: 0.5 * (1 + tanh(√(2/π) * (x + 0.044715·x³)))
#    discretized_gaussian_log_likelihood()에서 내부적으로 호출.
#
# 3. continuous_gaussian_log_likelihood(x, means, log_scales):
#    연속 Gaussian 분포의 log-likelihood 계산.
#    log N(x; μ, σ²) 를 반환. 표준 정규분포로 정규화 후 log_prob 계산.
#
# 4. discretized_gaussian_log_likelihood(x, means, log_scales):
#    이산화된(discretized) Gaussian 분포의 log-likelihood 계산.
#    이미지 픽셀이 uint8(0~255)에서 [-1,1]로 rescale된 것을 가정하고,
#    각 픽셀 bin의 확률을 CDF 차이로 계산한다:
#      P(x) = Φ((x+1/255-μ)/σ) - Φ((x-1/255-μ)/σ)
#
#    경계 처리:
#    - x < -0.999 (0에 가까운 픽셀): log Φ((x+1/255-μ)/σ) 사용
#    - x > 0.999 (255에 가까운 픽셀): log(1 - Φ((x-1/255-μ)/σ)) 사용
#    - 그 외: log(CDF 차이) 사용
#
#    사용처: _vb_terms_bpd()에서 t=0일 때(마지막 denoising step)의 decoder
#    negative log-likelihood 계산에 사용.

import torch as th
import numpy as np


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a continuous Gaussian distribution.
    :param x: the targets
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = th.distributions.Normal(th.zeros_like(x), th.ones_like(x)).log_prob(normalized_x)
    return log_probs


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
