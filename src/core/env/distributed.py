# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# 분산 학습 유틸리티 모음.
#
# 주요 기능:
#   - init_distributed(): torchrun 및 SLURM 환경을 지원하는 PyTorch 분산
#     프로세스 그룹 초기화.
#   - 헬퍼 함수: get_world_size(), get_rank(), is_main_process() 등.
#
import torch
import torch.distributed as dist

import os
import datetime
import builtins
from logging import getLogger

logger = getLogger()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def init_distributed(port=37124, rank_and_world_size=(None, None)):
    rank, world_size = rank_and_world_size
    dist_url='env://'
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(port))
    print("Using port", os.environ['MASTER_PORT'])

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            gpu = int(os.environ["LOCAL_RANK"])
        except Exception:
            logger.info('torchrun env vars not sets')

    elif "SLURM_PROCID" in os.environ:
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            gpu = rank % torch.cuda.device_count()
            if 'HOSTNAME' in os.environ:
                os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
            else:
                os.environ['MASTER_ADDR'] = '127.0.0.1'
        except Exception:
            logger.info('SLURM vars not set')
    
    else:
        rank = 0
        world_size = 1
        gpu = 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'

    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        init_method=dist_url
    )

    # setup_for_distributed(rank == 0)
    return world_size, rank, gpu, True
