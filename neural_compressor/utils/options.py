#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
"""ONNX options."""

from .utility import DotDict


class onnxrt:
    """ONNX helper configuration."""

    graph_optimization = DotDict({"level": None, "gemm2matmul": True})
    qdq_setting = DotDict(
        {"OpTypesToExcludeOutputQuantizatioin": [], "AddQDQPairToWeight": False, "DedicatedQDQPair": False}
    )


OPTIONS = {
    "tensorflow": None,
    "tensorflow_itex": None,
    "pytorch": None,
    "pytorch_fx": None,
    "pytorch_ipex": None,
    "mxnet": None,
    "onnxrt_integerops": onnxrt,
    "onnxrt_qlinearops": onnxrt,
    "onnxrt_qdq": onnxrt,
    "onnxruntime": onnxrt,
}
