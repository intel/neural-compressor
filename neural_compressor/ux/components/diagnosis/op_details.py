# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
"""The OpDetails class."""
from typing import Any, Dict, List

from neural_compressor.ux.utils.json_serializer import JsonSerializer


class OpDetails(JsonSerializer):
    """OP details class."""

    def __init__(self, op_name: str, op_details: dict) -> None:
        """Initialize OP details."""
        super().__init__()
        self.op_name: str = op_name
        self.pattern: OpPattern = OpPattern(op_details.get("pattern", {}))
        self.weight: OpWeights = OpWeights(op_details.get("weight", {}))
        self.activation: OpActivation = OpActivation(op_details.get("activation", {}))

    def serialize(
        self,
        serialization_type: str = "default",
    ) -> Dict[str, Any]:
        """Serialize OpDetails."""
        return {
            "OP name": self.op_name,
            "Pattern": self.pattern.serialize(),
            "Weights": self.weight.serialize(),
            "Activation": self.activation.serialize(),
        }


class OpPattern(JsonSerializer):
    """OP pattern class."""

    def __init__(self, pattern_data: dict):
        """Initialize OP pattern."""
        super().__init__()
        self.sequence: List[str] = pattern_data.get("sequence", "").split(",")
        self.precision: str = pattern_data.get("precision", None)


class OpWeights(JsonSerializer):
    """OP weights class."""

    def __init__(self, weights_data: dict):
        """Initialize OP weights."""
        super().__init__()
        self.dtype: str = weights_data.get("dtype", None)
        self.granularity: str = weights_data.get("granularity", None)


class OpActivation(JsonSerializer):
    """OP activation class."""

    def __init__(self, op_activation: dict):
        """Initialize OP activation."""
        super().__init__()
        self.dtype: str = op_activation.get("dtype", None)
