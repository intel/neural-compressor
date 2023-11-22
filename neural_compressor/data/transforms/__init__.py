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
# ==============================================================================
"""Neural Compressor Built-in transforms for multiple framework backends."""

from .transform import (
    TRANSFORMS,
    BaseTransform,
    ComposeTransform,
    transform_registry,
    ResizeTFTransform,
    TensorflowResizeWithRatio,
    RescaleTFTransform,
    NormalizeTFTransform,
)
from .transform import TFSquadV1PostTransform, TFSquadV1ModelZooPostTransform
from .coco_transform import ParseDecodeCocoTransform
from .postprocess import Postprocess
from .imagenet_transform import LabelShift, BilinearImagenetTransform, TensorflowResizeCropImagenetTransform
from .imagenet_transform import TensorflowShiftRescale
from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))

for f in modules:
    if isfile(f) and not f.startswith("__") and not f.endswith("__init__.py"):
        __import__(basename(f)[:-3], globals(), locals(), level=1)


__all__ = [
    "TRANSFORMS",
    "BaseTransform",
    "ComposeTransform",
    "transform_registry",
    "ResizeTFTransform",
    "Postprocess",
    "LabelShift",
    "BilinearImagenetTransform",
    "TensorflowResizeCropImagenetTransform",
    "RescaleTFTransform",
    "NormalizeTFTransform",
    "ParseDecodeCocoTransform",
    "TensorflowResizeWithRatio",
    "TFSquadV1PostTransform",
    "TFSquadV1ModelZooPostTransform",
    "TensorflowShiftRescale",
]
