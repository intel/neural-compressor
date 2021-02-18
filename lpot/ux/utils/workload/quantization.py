# -*- coding: utf-8 -*-
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
"""Configuration quantization module."""

from typing import Any, Dict, List, Union

from lpot.ux.utils.json_serializer import JsonSerializer
from lpot.ux.utils.workload.dataloader import Dataloader


class Calibration(JsonSerializer):
    """Configuration Calibration class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Calibration class."""
        super().__init__()
        self.sampling_size: Union[int, List[int], str] = data.get("sampling_size", 100)
        self.dataloader = None
        if isinstance(data.get("dataloader"), dict):
            self.dataloader = Dataloader(
                data=data.get("dataloader", {}),
                batch_size=1,
            )


class WiseConfigDetails(JsonSerializer):
    """Configuration WiseConfigDetails class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration WiseConfigDetails class."""
        super().__init__()
        # [Optional] One of "per_channel", "per_tensor"
        self.granularity = data.get("granularity", None)
        # [Optional] One of "asym", "sym"
        self.scheme = data.get("scheme", None)
        # [Optional] One of "int8", "uint8", "fp32", "bf16"
        self.dtype = data.get("dtype", None)
        # [Optional] One of "minmax", "kl"
        self.algorithm = data.get("algorithm", None)


class WiseConfig(JsonSerializer):
    """Configuration WiseConfig class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration WiseConfig class."""
        super().__init__()
        self.weight = None
        if isinstance(data.get("weight"), dict):
            self.weight = WiseConfigDetails(data.get("weight", {}))  # [Optional]

        self.activation = None
        if isinstance(data.get("activation"), dict):
            self.activation = WiseConfigDetails(
                data.get("activation", {}),
            )  # [Optional]


class Advance(JsonSerializer):
    """Configuration Advance class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Advance class."""
        super().__init__()
        self.bias_correction = data.get("bias_correction", None)


class Quantization(JsonSerializer):
    """Configuration Quantization class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Quantization class."""
        super().__init__()
        self.calibration = None
        if isinstance(data.get("calibration"), dict):
            self.calibration = Calibration(data.get("calibration", {}))

        self.model_wise = None
        if isinstance(data.get("model_wise"), dict):
            self.model_wise = WiseConfig(data.get("model_wise", {}))

        self.op_wise = None

        if isinstance(data.get("op_wise"), dict):
            self.op_wise = data.get("op_wise", {})

        self.approach = data.get("approach", "post_training_static_quant")

        self.advance = None
        if isinstance(data.get("advance"), dict):
            self.advance = Advance(data.get("advance", {}))
