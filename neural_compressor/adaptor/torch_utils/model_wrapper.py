#
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

"""Torch.nn.Module Class Defination."""
# Note: Do not import this file unless you have already imported torch, 
# since the model classes inherit torch.nn.Module.
import torch
from packaging.version import Version


def get_torch_version():
    try:
        torch_version = torch.__version__.split('+')[0]
    except ValueError as e:  # pragma: no cover
        assert False, 'Got an unknown version of torch: {}'.format(e)
    version = Version(torch_version)
    return version

PT_VERSION = get_torch_version().release


class QDQLinear(torch.nn.Module):
    def __init__(self, module, scale, zero_point, dtype):
        super().__init__()
        if PT_VERSION < Version("1.13.0").release:
            import torch.nn.quantized as nnq
        else:
            import torch.ao.nn.quantized as nnq
        self.add_module('quant', nnq.Quantize(scale, zero_point, dtype))
        self.add_module('dequant', nnq.DeQuantize())
        self.add_module('module', module)
        self.qdq_weight()

    def forward(self, X):
        X = self.quant(X)
        X = self.dequant(X)
        X = self.module(X)
        return X

    def qdq_weight(self):
        # update weight w/ QDQ
        from .smooth_quant import quant_dequant_w
        weith_qdq = quant_dequant_w(self.module)
        self.module.weight = torch.nn.Parameter(weith_qdq)


class SQLinearWrapper(torch.nn.Module):
    def __init__(self, module, input_scale, input_minmax, dtype=torch.quint8):
        super().__init__()
        self.input_scale = input_scale
        self.dtype = dtype
        # calculate and only save scale, zero_point to avoid memory usage
        self.scale, self.zero_point = self._calculate_qparams(input_scale, input_minmax, dtype)
        self.add_module('sq_linear', module)
        self.ipex = False  # a flag used for ipex inference

    def forward(self, X):
        if self.ipex:
            X = self.sq_linear(X)
        else:
            X = torch.mul(X, self.input_scale)
            X = self.sq_linear(X)
        return X

    def _calculate_qparams(self, input_scale, input_minmax, dtype=torch.quint8):
        # calculate scale and zero_point
        if dtype == torch.quint8:
            quant_min, quant_max = 0, 255
        min_val = torch.min(input_minmax[0] * input_scale)
        max_val = torch.max(input_minmax[1] * input_scale)
        # work when min_val bigger than zero.
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, torch.tensor([torch.finfo(torch.float32).eps]))
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def _get_weight_scale(self):
        # get weight scale and zero_point
        from torch.ao.quantization.observer import default_per_channel_weight_observer
        obs = default_per_channel_weight_observer()
        obs(self.sq_linear.weight)
        scale, _ = obs.calculate_qparams()
        return scale

    def _recover_sq_linear(self):
        # remove mul and reset sq_linear for ipex inference
        scale = self.input_scale.view(1, self.input_scale.shape[0])
        with torch.no_grad():
            self.sq_linear.weight *= scale
