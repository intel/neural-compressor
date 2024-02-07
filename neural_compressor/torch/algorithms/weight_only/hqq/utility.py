# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gc
import time

import numpy as np
import psutil
import torch

from neural_compressor.torch.utils import logger

__all__ = [
    "is_divisible",
    "dump_elapsed_time",
]


def is_divisible(val1, val2):
    return int(val2 * np.ceil(val1 / val2)) == val1


def see_cuda_memory_usage(message, force=False):  # pragma: no cover
    # Copied from https://github.com/microsoft/DeepSpeed
    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # logger.info message except when distributed but not rank 0
    logger.info(message)
    logger.info(
        f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024))} GB "
    )
    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logger.info(f"CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%")

    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()


def dump_elapsed_time(customized_msg=""):
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): The parameter passed to decorator. Defaults to None.
    """

    def f(func):
        def fi(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            logger.info(
                "%s elapsed time: %s ms"
                % (customized_msg if customized_msg else func.__qualname__, round((end - start) * 1000, 2))
            )
            return res

        return fi

    return f
