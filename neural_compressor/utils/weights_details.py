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
"""The Weights details class."""
from typing import Any, Dict, Optional

import numpy as np

from neural_compressor.utils.utility import mse_metric_gap

PRECISION = 5


class WeightsDetails:
    """Weights details class."""

    def __init__(self, op_name: str, input_tensor_data, optimized_tensor_data):
        """Initialize Weights details."""
        self.op_name: str = op_name
        self.mse: Optional[float] = mse_metric_gap(
            input_tensor_data,
            optimized_tensor_data,
        )
        self.input_stats: WeightsStatistics = WeightsStatistics(input_tensor_data)
        self.optimized_stats: WeightsStatistics = WeightsStatistics(optimized_tensor_data)

    def serialize(self) -> Dict[str, Any]:
        """Serialize Weights details."""
        return {
            "OP name": self.op_name,
            "MSE": self.mse,
            "Input model": self.input_stats.serialize(),
            "Optimized model": self.optimized_stats.serialize(),
        }


class WeightsStatistics:
    """Weights statistics class."""

    def __init__(self, tensor_data) -> None:
        """Initialize Weights details.

        Args:
            tensor_data: numpy array with tensor data

        Returns:
            None
        """
        self.min: float = np.min(tensor_data)
        self.max: float = np.max(tensor_data)
        self.mean: float = np.mean(tensor_data)
        self.std: float = np.std(tensor_data)
        self.var: float = np.var(tensor_data)

    def serialize(self) -> Dict[str, Any]:
        """Serialize Weights details.

        Returns:
            Dictionary with serialized WeightsStatistics object
        """
        return {
            "Min weight": round(self.min, PRECISION),
            "Max weight": round(self.max, PRECISION),
            "Mean": round(self.mean, PRECISION),
            "Standard deviation": round(self.std, PRECISION),
            "Variance": round(self.var, PRECISION),
        }
