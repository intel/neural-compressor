#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

from neural_compressor.common import logger

__all__ = [
    "set_workspace",
    "set_random_seed",
    "set_resume_from",
    "set_tensorboard",
    "dump_elapsed_time",
]


def set_random_seed(seed: int):
    """Set the random seed in config."""
    from neural_compressor.common import options

    options.random_seed = seed


def set_workspace(workspace: str):
    """Set the workspace in config."""
    from neural_compressor.common import options

    options.workspace = workspace


def set_resume_from(resume_from: str):
    """Set the resume_from in config."""
    from neural_compressor.common import options

    options.resume_from = resume_from


def set_tensorboard(tensorboard: bool):
    """Set the tensorboard in config."""
    from neural_compressor.common import options

    options.tensorboard = tensorboard


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
