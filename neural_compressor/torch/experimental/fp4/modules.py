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
        self.weight = torch.nn.Parameter(qdq_weight)

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 1:
            x = x.view(1, -1)
            qdq_x = full_quant(x)[0].reshape(x_shape)
        else:
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

    def forward(self, x, offsets=None, per_sample_weights=None):
        x_shape = x.shape
        if len(x_shape) == 1:
            x = x.view(1, -1)
            qdq_x = full_quant(x)[0].reshape(x_shape)
        else:
            qdq_x = full_quant(x)[0]
        return torch.nn.functional.embedding_bag(
            qdq_x, self.weight, offsets=offsets, mode=self.mode,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse, per_sample_weights=per_sample_weights
        )

def qdq_fp8(x, dtype=torch.float8_e4m3fn):
    if dtype == torch.float8_e4m3fn and x.abs().max() > 448:
        raise ValueError("FP8 quantization is only supported for values in the range [-448, 448].")
    return x.to(dtype).to(torch.float32)

class FP8EmbeddingBag(torch.nn.Module):
    def __init__(self, orig_layer):
        super().__init__()
        # migrate attributes from the original layer
        self.__dict__.update(orig_layer.__dict__)
        self.extra_repr = orig_layer.extra_repr
        # qdq weight
        qdq_weight = qdq_fp8(orig_layer.weight)
        self.weight = torch.nn.Parameter(qdq_weight)

    def forward(self, x, offsets=None, per_sample_weights=None):
        x_shape = x.shape
        if len(x_shape) == 1:
            x = x.view(1, -1)
            qdq_x = qdq_fp8(x).reshape(x_shape)
        else:
            qdq_x = qdq_fp8(x)
        return torch.nn.functional.embedding_bag(
            qdq_x, self.weight, offsets=offsets, mode=self.mode,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse, per_sample_weights=per_sample_weights
        )
