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
"""Configuration evaluation module."""

from typing import Any, Dict, List, Optional, Union

from neural_compressor.ux.utils.hw_info import HWInfo
from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.utils.workload.dataloader import Dataloader


class Metric(JsonSerializer):
    """Configuration Metric class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Metric class."""
        super().__init__()
        self._name: str = data.get("name", None)
        self._param: Optional[Union[int, bool]] = data.get("param", None)

        if len(data) == 1 and not (self.name and self.param):
            self.name = list(data.keys())[0]
            self.param = data.get(self.name, None)

    @property
    def name(self) -> str:
        """Get metric name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set metric name."""
        self._name = value

    @property
    def param(self) -> Optional[Union[int, bool]]:
        """Get metric param."""
        return self._param

    @param.setter
    def param(self, value: Union[int, bool]) -> None:
        """Set metric param."""
        if not (isinstance(value, str) and value == ""):
            self._param = value

    def serialize(
        self,
        serialization_type: str = "default",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Return metric dict for config."""
        if self.param and isinstance(self.param, dict):
            return {self.name: self.param}
        if self.name in ["MSE", "RMSE", "MAE"]:
            return {self.name: {"compare_label": self.param}}
        if self.name in ["COCOmAP"] and self.param:
            return {self.name: {"anno_path": self.param}}
        if self.name:
            return {self.name: self.param}
        return {}


class Configs(JsonSerializer):
    """Configuration Configs class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Configs class."""
        super().__init__()

        self.cores_per_instance: int = data.get("cores_per_instance", 4)
        self.num_of_instance: int = data.get(
            "num_of_instance",
            HWInfo().cores // self.cores_per_instance,
        )
        self.inter_num_of_threads: int = data.get("inter_num_of_threads", None)
        self.intra_num_of_threads: int = data.get("intra_num_of_threads", None)
        self.kmp_blocktime: int = data.get("kmp_blocktime", 1)


class PostprocessSchema(JsonSerializer):
    """Configuration PostprocessSchema class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration PostprocessSchema class."""
        super().__init__()
        self.LabelShift = data.get("LabelShift", None)  # [Optional] >0
        self.SquadV1 = data.get("SquadV1", None)


class Postprocess(JsonSerializer):
    """Configuration Postprocess class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Postprocess class."""
        super().__init__()
        self.transform = None
        if isinstance(data.get("transform"), dict):
            self.transform = PostprocessSchema(data.get("transform", {}))


class Accuracy(JsonSerializer):
    """Configuration Accuracy class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Accuracy class."""
        super().__init__()
        self.metric = None
        if isinstance(data.get("metric"), dict):
            self.metric = Metric(data.get("metric", {}))

        self.configs = None
        if isinstance(data.get("configs"), dict):
            self.configs = Configs(data.get("configs", {}))

        self.dataloader = None
        if isinstance(data.get("dataloader"), dict):
            self.dataloader = Dataloader(data.get("dataloader", {}))

        self.postprocess = None
        if isinstance(data.get("postprocess"), dict):
            self.postprocess = Postprocess(data.get("postprocess", {}))


class Performance(JsonSerializer):
    """Configuration Performance class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Performance class."""
        super().__init__()
        self.warmup: int = data.get("warmup", 5)

        self.iteration: int = data.get("iteration", -1)

        self.configs: Configs = Configs(data.get("configs", {}))

        self.dataloader: Optional[Dataloader] = None
        if isinstance(data.get("dataloader"), dict):
            self.dataloader = Dataloader(data.get("dataloader", {}))

        self.postprocess: Optional[Postprocess] = None
        if isinstance(data.get("postprocess"), dict):
            self.postprocess = Postprocess(data.get("postprocess", {}))


class Evaluation(JsonSerializer):
    """Configuration Evaluation class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Evaluation class."""
        super().__init__()
        self.accuracy: Optional[Accuracy] = None
        if isinstance(data.get("accuracy"), dict):
            self.accuracy = Accuracy(data.get("accuracy", {}))

        self.performance = None
        if isinstance(data.get("performance"), dict):
            self.performance = Performance(data.get("performance", {}))
