# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn as nn

from helpers import Optimization
from parts.features import FeatureFactory


class AudioPreprocessing(nn.Module):
    """GPU accelerated audio preprocessing
    """

    def __init__(self, **kwargs):
        nn.Module.__init__(self)    # For PyTorch API
        self.optim_level = kwargs.get(
            'optimization_level', Optimization.nothing)
        self.featurizer = FeatureFactory.from_config(kwargs)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_signal, length = x
        length.requires_grad_(False)
        processed_signal = self.featurizer(x)
        processed_length = self.featurizer.get_seq_len(length)
        return processed_signal, processed_length
