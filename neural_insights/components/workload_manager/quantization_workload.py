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
"""The quantization_workload module for Neural Insights quantization workloads."""
from typing import Optional, Dict, Any

from neural_insights.components.workload_manager.workload import Workload
from neural_insights.utils.json_serializer import JsonSerializer


class QuantizationWorkload(Workload):
    """QuantizationWorkload class."""

    def __init__(self, data: Optional[dict] = None):
        """Initialize Quantization Workload."""
        super().__init__(data)
        if data is None:
            data = {}
        self.accuracy_data = AccuracyData(data.get("accuracy_data", None))

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize Workload class."""
        serialized_workload = super().serialize(serialization_type)
        serialized_workload.update({"accuracy_data": self.accuracy_data.serialize()})
        return serialized_workload


class AccuracyData(JsonSerializer):
    """AccuracyData class."""

    def __init__(self, data: Optional[dict] = None):
        """Initialize Accuracy Data."""
        super().__init__()
        if data is None:
            data = {}
        self._baseline_accuracy: Optional[float] = data.get("baseline_accuracy", None)
        self._optimized_accuracy: Optional[float] = data.get("optimized_accuracy", None)

    @property
    def baseline_accuracy(self) -> Optional[float]:
        """Get baseline accuracy.

        Returns: baseline accuracy value
        """
        return self._baseline_accuracy

    @property
    def optimized_accuracy(self) -> Optional[float]:
        """Get optimized accuracy.

        Returns: optimized accuracy value
        """
        return self._optimized_accuracy

    @property
    def ratio(self) -> Optional[float]:
        """Get accuracy ratio.

        Returns: accuracy ratio if baseline and optimized accuracy are present
                 Otherwise returns None
        """
        if self.optimized_accuracy in [None, 0] or self.baseline_accuracy in [None, 0]:
            return None

        return (self.optimized_accuracy - self.baseline_accuracy) / self.baseline_accuracy

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize AccuracyData class."""
        return {
            "baseline_accuracy": self.baseline_accuracy,
            "optimized_accuracy": self.optimized_accuracy,
            "ratio": self.ratio,
        }
