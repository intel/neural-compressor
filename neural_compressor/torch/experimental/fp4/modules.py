# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from auto_round.data_type.nvfp import full_quant

# "NVFP" stands for NVIDIA FP4, which is an accepted abbreviation in this context.
class NVFP4Linear(torch.nn.Module):
    def __init__(self, orig_linear):
        super().__init__()
        self.__dict__.update(orig_linear.__dict__)
        qdq_weight = full_quant(orig_linear.weight)
        self.weight = torch.nn.Parameter(qdq_weight.t().contiguous())

    def forward(self, x):
        qdq_x = full_quant(x)
        return torch.nn.functional.linear(qdq_x, self.weight, self.bias)
