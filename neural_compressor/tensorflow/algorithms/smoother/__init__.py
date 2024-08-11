#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""Initialize classes of smooth quant algorithm."""

from neural_compressor.tensorflow.algorithms.smoother.core import SmoothQuant
from neural_compressor.tensorflow.algorithms.smoother.scaler import (
    SmoothQuantScaler,
    SmoothQuantScalerLLM,
)
from neural_compressor.tensorflow.algorithms.smoother.calibration import (
    SmoothQuantCalibration,
    SmoothQuantCalibrationLLM,
)
