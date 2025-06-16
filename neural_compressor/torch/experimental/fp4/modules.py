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
import numpy as np
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


def qdq_fp8(x, dtype=torch.float8_e4m3fn, scale_format="opt", per_channel=True):
    """
    Quantize and dequantize a tensor using FP8 format.
    Args:
        x (torch.Tensor): Input tensor to be quantized and dequantized.
        dtype (torch.dtype): The FP8 data type to use for quantization.
        scale_format (str): The format of the scale, either "pow2" or "raw".
        per_channel (bool): If True, compute scale per channel; otherwise, compute a single scale for the entire tensor.
    Returns:
        tuple: A tuple containing the quantized and dequantized tensor, and the scale used for quantization.
    """
    orig_dtype = x.dtype
    assert dtype in [torch.float8_e4m3fn, torch.float8_e5m2], "Only FP8 quantization is supported."
    FP8_MAX = 448 if dtype == torch.float8_e4m3fn else 57344
    if per_channel:
        scale = torch.abs(x).max(dim=-1, keepdim=True)[0] / FP8_MAX
    else:
        scale = torch.abs(x).max() / FP8_MAX
    if scale_format == "pow2":
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    elif scale_format == "opt":
        raw_scale = scale.clone()
        best_mse = float('inf')
        best_scale = scale
        for i in np.arange(1.0, 2.0, 0.1):
            scale = raw_scale * i
            q_x = (x / scale).to(dtype)
            qdq_x = q_x.to(torch.float32) * scale
            mse = torch.mean((qdq_x - x) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_scale = scale
        scale = best_scale

    q_x = (x / scale).to(dtype)
    qdq_x = q_x.to(torch.float32) * scale
    return qdq_x.to(orig_dtype), scale


class FP8EmbeddingBag(torch.nn.Module):
    def __init__(self, orig_layer):
        super().__init__()
        # migrate attributes from the original layer
        self.__dict__.update(orig_layer.__dict__)
        self.extra_repr = orig_layer.extra_repr
        # qdq weight
        qdq_weight = qdq_fp8(orig_layer.weight)[0]
        self.weight = torch.nn.Parameter(qdq_weight)

    def forward(self, x, offsets=None, per_sample_weights=None):
        x_shape = x.shape
        if len(x_shape) == 1:
            x = x.view(1, -1)
            qdq_x = qdq_fp8(x)[0].reshape(x_shape)
        else:
            qdq_x = qdq_fp8(x)[0]
        return torch.nn.functional.embedding_bag(
            qdq_x, self.weight, offsets=offsets, mode=self.mode,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse, per_sample_weights=per_sample_weights
        )
