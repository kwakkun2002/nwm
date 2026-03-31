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
# 학습 시 diffusion timestep을 샘플링하는 전략을 정의하는 모듈.
#
# Diffusion 모델 학습에서는 매 배치마다 랜덤 timestep t를 선택하여 해당 노이즈 수준에서의
# 손실을 계산한다. 이 모듈은 timestep 샘플링 분포를 제어하여 학습 효율을 조절한다.
#
# ── 주요 구성 요소 ──
#
# 1. ScheduleSampler (추상 기반 클래스):
#    모든 timestep sampler의 부모 클래스. Importance sampling 프레임워크를 제공.
#
#    - weights(): 각 timestep의 샘플링 가중치를 반환 (서브클래스에서 구현)
#    - sample(batch_size, device):
#      가중치를 확률로 정규화한 후 np.random.choice()로 timestep을 샘플링.
#      Importance sampling 보정을 위해 역확률 가중치(1/(T·p_t))도 함께 반환.
#      이 가중치를 손실에 곱하면 기대값이 uniform sampling과 동일해진다.
#
#      반환: (timestep 인덱스 텐서, importance weight 텐서)
#
# 2. UniformSampler(ScheduleSampler):
#    모든 timestep에 동일한 가중치(=1)를 부여하는 균등 샘플러.
#    가장 단순한 전략으로, 모든 timestep이 동일한 확률로 선택된다.
#    NWM 프로젝트의 학습에서 기본적으로 사용됨.
#
# 3. LossAwareSampler(ScheduleSampler):
#    손실 값에 기반하여 timestep 샘플링 확률을 동적으로 조절하는 추상 클래스.
#
#    - update_with_local_losses(local_ts, local_losses):
#      분산 학습 환경에서 각 rank의 로컬 손실을 all_gather로 수집하여 동기화.
#      모든 rank가 동일한 상태를 유지하도록 보장.
#    - update_with_all_losses(): 서브클래스에서 구현, 전체 손실로 가중치 갱신.
#
# 4. LossSecondMomentResampler(LossAwareSampler):
#    Improved DDPM(Nichol & Dhariwal, 2021)의 importance sampling 전략 구현.
#
#    각 timestep별로 최근 history_per_term(기본 10)개의 손실 이력을 유지하고,
#    손실의 이차 모멘트(second moment) √(E[L²])에 비례하여 샘플링 확률을 설정.
#    → 손실이 크고 분산이 높은 timestep이 더 자주 샘플링되어 학습 분산을 줄임.
#
#    uniform_prob(기본 0.001)를 혼합하여 모든 timestep이 최소한의 확률을 가지도록 보장.
#    모든 timestep에 대해 history_per_term개의 손실이 수집될 때까지(_warmed_up)는
#    uniform sampling을 사용.
#
# ── 팩토리 함수 ──
#
# create_named_schedule_sampler(name, diffusion):
#   이름으로 sampler를 생성.
#   - "uniform": UniformSampler
#   - "loss-second-moment": LossSecondMomentResampler

from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.
    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.
    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.
        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.
        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.
        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.
        Sub-classes should override this method to update the reweighting
        using losses from the model.
        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.
        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
