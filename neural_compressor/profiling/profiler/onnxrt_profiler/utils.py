# -*- coding: utf-8 -*-
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
"""ONNX profiler utils."""

from typing import Any


def create_onnx_config(ort: Any, intra_num_of_threads: int, inter_num_of_threads: int) -> Any:
    """Create tensorflow config.

    Args:
        ort: onnxruntime library
        intra_num_of_threads: number of threads used within an individual op for parallelism
        inter_num_of_threads: number of threads used for parallelism between independent operations

    Returns:
        ONNX SessionOptions object
    """
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.intra_op_num_threads = intra_num_of_threads
    options.inter_op_num_threads = inter_num_of_threads
    return options
