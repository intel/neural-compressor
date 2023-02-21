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

"""Build SmoothQuant algorithm class."""

import numpy as np
from .algorithm import Algorithm, algorithm_registry
from ..utils import logger

@algorithm_registry(algorithm_type='smooth_quant')
class SmoothQuant(Algorithm):
    """SmoothQuant algorithm class."""
    def __init__(self, percentile=99.999, op_types=['MatMul', 'Linear', 'Conv'],
                             scales_per_op=True, alpha=1.0):
        """Initialize SmoothQuant class.

        Args:
            percentile:Percentile of calibration to remove outliers
            op_types: The op types whose input tensor will be dumped
            scales_per_op: True, each op will have an individual scale, mainly for accuracy
                           False, ops with the same input will share a scale, mainly for performance
            alpha: migrate strength.
        """
        self.percentile = percentile
        self.op_types = op_types
        self.scales_per_op = scales_per_op
        self.alpha = alpha
        self.tune_cfg = None

    def __call__(self, origin_model, q_model, adaptor, dataloader, iterations):
        """Return the processed model via SmoothQuant algorithm.

        Fake input channel quantization, for more details please refer to:
        [1] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
        [2] SPIQ: Data-Free Per-Channel Static Input Quantization
        inert Mul op before each conv/matmul with adjusted weights

        Args:
            origin_model: origin_model
            q_model: q_model
            adaptor: adaptor
            dataloader: dataloader
            iterations: iterations

        Returns:
            model: A modified onnx model
        """
        q_model = adaptor.smooth_quant(origin_model, dataloader, iterations, self.tune_cfg, self.alpha,
                                        self.percentile, self.op_types, self.scales_per_op)
        return q_model