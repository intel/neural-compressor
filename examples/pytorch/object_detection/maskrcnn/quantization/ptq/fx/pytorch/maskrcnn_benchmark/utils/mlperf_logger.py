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

import os
import torch

from maskrcnn_benchmark.utils.comm import get_rank, is_main_process
from mlperf_logging import mllog
from mlperf_logging.mllog import constants

mllogger = mllog.get_mllogger()

def log_start(*args, **kwargs):
    _log_print(mllogger.start, *args, **kwargs)
def log_end(*args, **kwargs):
    _log_print(mllogger.end, *args, **kwargs)
def log_event(*args, **kwargs):
    _log_print(mllogger.event, *args, **kwargs)
def _log_print(logger, *args, **kwargs):
    """
    Wrapper for MLPerf compliance logging calls.
    All arguments but 'log_all_ranks' are passed to
    mlperf_logging.mllog.
    If 'log_all_ranks' is set to True then all distributed workers will print
    logging message, if set to False then only worker with rank=0 will print
    the message.
    """
    if 'stack_offset' not in kwargs:
        kwargs['stack_offset'] = 3
    if 'value' not in kwargs:
        kwargs['value'] = None

    if kwargs.pop('log_all_ranks', False):
        log = True
    else:
        log = (get_rank() == 0)

    if log:
        logger(*args, **kwargs)

def configure_logger(benchmark):
    mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{benchmark}.log'))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False

def mlperf_submission_log(benchmark):
    required_dist_init = ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']

    if all(var in os.environ for var in required_dist_init):
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    num_nodes = os.environ.get('SLURM_NNODES', 1)

    configure_logger(benchmark)

    log_event(
        key=constants.SUBMISSION_BENCHMARK,
        value=benchmark,
        )

    log_event(
        key=constants.SUBMISSION_ORG,
        value='NVIDIA')

    log_event(
        key=constants.SUBMISSION_DIVISION,
        value='closed')

    log_event(
        key=constants.SUBMISSION_STATUS,
        value='onprem')

    log_event(
        key=constants.SUBMISSION_PLATFORM,
        value=f'{num_nodes}xSUBMISSION_PLATFORM_PLACEHOLDER')

def generate_seeds(rng, size):
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds

def broadcast_seeds(seeds, device):
    if torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor(seeds).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds
