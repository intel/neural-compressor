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
"""Half-precision Wrapper for Torch Modules."""

import torch


class HalfPrecisionModuleWrapper(torch.nn.Module):
    """FP16 or BF16 Module Wrapper Class."""

    def __init__(self, module, device="cpu", dtype=torch.float16):
        """Init a HalfPrecisionModuleWrapper object."""
        super(HalfPrecisionModuleWrapper, self).__init__()
        self.add_module("module", module)
        self.device = device
        self.dtype = dtype
        self.weight = self.module.weight if hasattr(self.module, "weight") else None
        self.bias = self.module.bias if hasattr(self.module, "bias") else None

    def forward(self, X):
        """Convert dtype."""
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            X = self.module(X)
        return X.float()
