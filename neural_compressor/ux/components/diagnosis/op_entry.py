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
"""The OpEntry class."""
from typing import Any, Dict

from neural_compressor.ux.utils.json_serializer import JsonSerializer


class OpEntry(JsonSerializer):
    """OP entry class."""

    def __init__(self, op_name: str, mse: float, activation_min: float, activation_max: float):
        """Initialize OP entry."""
        super().__init__()
        self.op_name: str = op_name
        self.mse: float = mse
        self.activation_min: float = activation_min
        self.activation_max: float = activation_max

    def serialize(
        self,
        serialization_type: str = "default",
    ) -> Dict[str, Any]:
        """Serialize OP entry."""
        return {
            "OP name": self.op_name,
            "MSE": self.mse,
            "Activation Min": self.activation_min,
            "Activation Max": self.activation_max,
        }
