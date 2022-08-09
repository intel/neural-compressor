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

"""Constant values."""

import os
from enum import Enum

GITHUB_ORG = "intel"
GITHUB_REPO = "neural-compressor"
WORKDIR_LOCATION = os.path.join(os.environ.get("HOME", ""), ".neural_compressor")
WORKSPACE_LOCATION = os.path.join(os.environ.get("HOME", ""), "workdir")


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


class OptimizationTypes(Enum):
    """OptimizationTypes enumeration."""

    QUANTIZATION = "Quantization"
    GRAPH_OPTIMIZATION = "Graph optimization"


class ExecutionStatus(Enum):
    """Executions statuses for optimizations, benchmarks and profilings."""

    WIP = "wip"
    SUCCESS = "success"
    ERROR = "error"


class Precisions(Enum):
    """Precisions enumeration."""

    FP32 = "fp32"
    BF16 = "bf16"
    INT8 = "int8"
    INT8_STATIC_QUANTIZATION = "int8 static quantization"
    INT8_DYNAMIC_QUANTIZATION = "int8 dynamic quantization"


class Strategies(Enum):
    """Strategies enumeration."""

    BASIC = "basic"
    BAYESIAN = "bayesian"
    EXHAUSTIVE = "exhaustive"
    MSE = "mse"
    RANDOM = "random"
    SIGOPT = "sigopt"
    TPE = "tpe"


precision_optimization_types = {
    OptimizationTypes.QUANTIZATION: [
        Precisions.INT8,
        Precisions.INT8_STATIC_QUANTIZATION,
        Precisions.INT8_DYNAMIC_QUANTIZATION,
    ],
    OptimizationTypes.GRAPH_OPTIMIZATION: [Precisions.BF16, Precisions.FP32],
}

postprocess_transforms = ["SquadV1"]
