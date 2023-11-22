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
"""Constant values."""

import os
from enum import Enum

WORKDIR_LOCATION = os.path.join(os.environ.get("HOME", ""), ".neural_compressor")

CONFIG_REFRESH_PERIOD = 5


class WorkloadModes(Enum):
    """Workload mode enumeration."""

    BENCHMARK = "benchmark"
    QUANTIZATION = "quantization"


class Domains(Enum):
    """Model domains enumeration."""

    NONE = ""
    IMAGE_RECOGNITION = "Image Recognition"
    OBJECT_DETECTION = "Object Detection"
    NLP = "Neural Language Processing"
    RECOMMENDATION = "Recommendation"


class DomainFlavours(Enum):
    """Domain flavours enumeration."""

    NONE = ""
    SSD = "SSD"
    YOLO = "Yolo"


class Frameworks(Enum):
    """Supported frameworks enumeration."""

    TF = "TensorFlow"
    ONNX = "ONNXRT"
    PT = "PyTorch"


class WorkloadStatus(Enum):
    """Workload status enumeration."""

    NEW = "new"
    WIP = "wip"
    SUCCESS = "success"
    FAILURE = "failure"
