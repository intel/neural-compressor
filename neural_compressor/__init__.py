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
"""Intel® Neural Compressor: An open-source Python library supporting popular model compression techniques."""
from .version import __version__

# we need to set a global 'NA' backend, or Model can't be used
from .config import (
    DistillationConfig,
    PostTrainingQuantConfig,
    WeightPruningConfig,
    QuantizationAwareTrainingConfig,
    MixedPrecisionConfig,
)
from .contrib import *
from .model import *
from .metric import *
from .utils import options
from .utils.utility import set_random_seed, set_tensorboard, set_workspace, set_resume_from
