#
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
"""FP16 Wrapper for Torch Modules."""

import torch

class FP16ModuleWrapper(torch.nn.Module):
    """FP16Module Wrapper Class."""

    def __init__(self, module, device):
        """Init a FP16ModuleWrapper object."""
        super(FP16ModuleWrapper, self).__init__()
        self.add_module("module", module)
        self.train(module.training)
        self.device = device
        # WA for TransformerEncoder to access its Linear's weights and bias
        if isinstance(module, torch.nn.Linear):
            self.weight = self.module.weight if hasattr(self.module, "weight") else None
            self.bias = self.module.bias if hasattr(self.module, "bias") else None

    def forward(self, X):
        """Convert dtype."""
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            X = self.module(X)
        return X.float()
