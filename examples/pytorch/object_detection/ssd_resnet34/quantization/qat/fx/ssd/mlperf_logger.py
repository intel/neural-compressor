# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import os
from mlperf_logging import mllog
from mlperf_logging.mllog import constants as mllog_const
mllogger = mllog.get_mllogger()
mllog.config(
    filename=(os.getenv("COMPLIANCE_FILE") or "mlperf_compliance.log"),
    root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))))

def ssd_print(*args, sync=True, **kwargs):
    if sync:
        barrier()
    if get_rank() == 0:
        kwargs['stack_offset'] = 2
        mllogger.event(*args, **kwargs)


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        torch.cuda.synchronize()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = os.getenv('RANK', os.getenv('LOCAL_RANK', 0))
    return rank

def broadcast_seeds(seed, device):
    if torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor([seed]).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seed = seeds_tensor.item()
    return seed
