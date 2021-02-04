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
"""LPOT configuration evaluation module."""

from typing import Any, Dict

import psutil

from lpot.ux.utils.json_serializer import JsonSerializer
from lpot.ux.utils.workload.dataloader import Dataloader


class Metric(JsonSerializer):
    """LPOT configuration Metric class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize LPOT configuration Metric class."""
        super().__init__()
        self.topk = data.get("topk", 1)  # [Optional] One of 1, 5
        self.COCOmAP = data.get("COCOmAP", None)


class Configs(JsonSerializer):
    """LPOT configuration Configs class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize LPOT configuration Configs class."""
        super().__init__()
        self.cores_per_instance: int = 4
        self.num_of_instance: int = psutil.cpu_count() // self.cores_per_instance
        self.inter_num_of_threads: int = 4
        self.intra_num_of_threads: int = data.get("intra_num_of_threads", None)
        self.kmp_blocktime: int = 1


class PostprocessSchema(JsonSerializer):
    """LPOT configuration PostprocessSchema class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize LPOT configuration PostprocessSchema class."""
        super().__init__()
        self.LabelShift = data.get("LabelShift", None)  # [Optional] >0


class Postprocess(JsonSerializer):
    """LPOT configuration Postprocess class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize LPOT configuration Postprocess class."""
        super().__init__()
        self.transform = None
        if isinstance(data.get("transform"), dict):
            self.transform = PostprocessSchema(data.get("transform", {}))


class Accuracy(JsonSerializer):
    """LPOT configuration Accuracy class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize LPOT configuration Accuracy class."""
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
    """LPOT configuration Performance class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize LPOT configuration Performance class."""
        super().__init__()
        self.warmup = data.get("warmup", 10)

        self.iteration = data.get("iteration", -1)

        self.configs = Configs(data.get("configs", {}))

        self.dataloader = None
        if isinstance(data.get("dataloader"), dict):
            self.dataloader = Dataloader(data.get("dataloader", {}))

        self.postprocess = None
        if isinstance(data.get("postprocess"), dict):
            self.postprocess = Postprocess(data.get("postprocess", {}))


class Evaluation(JsonSerializer):
    """LPOT configuration Evaluation class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize LPOT configuration Evaluation class."""
        super().__init__()
        self.accuracy = None
        if isinstance(data.get("accuracy"), dict):
            self.accuracy = Accuracy(data.get("accuracy", {}))

        self.performance = None
        if isinstance(data.get("performance"), dict):
            self.performance = Performance(data.get("performance", {}))
