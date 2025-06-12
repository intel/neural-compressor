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
    def __init__(self, orig_layer):
        super().__init__()
        # migrate attributes from the original layer
        self.__dict__.update(orig_layer.__dict__)
        self.extra_repr = orig_layer.extra_repr
        # qdq weight
        qdq_weight = full_quant(orig_layer.weight)[0]
        self.weight = torch.nn.Parameter(qdq_weight.t().contiguous())

    def forward(self, x):
        qdq_x = full_quant(x)[0]
        return torch.nn.functional.linear(qdq_x, self.weight, self.bias)

class NVFP4EmbeddingBag(torch.nn.Module):
    def __init__(self, orig_layer):
        super().__init__()
        # migrate attributes from the original layer
        self.__dict__.update(orig_layer.__dict__)
        self.extra_repr = orig_layer.extra_repr
        # qdq weight
        qdq_weight = full_quant(orig_layer.weight)[0]
        self.weight = torch.nn.Parameter(qdq_weight)

    def forward(self, input, offsets=None, per_sample_weights=None):
        qdq_input = full_quant(input)[0]
        return torch.nn.functional.embedding_bag(
            qdq_input, self.weight, offsets=offsets, mode=self.mode,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse, per_sample_weights=per_sample_weights
        )
