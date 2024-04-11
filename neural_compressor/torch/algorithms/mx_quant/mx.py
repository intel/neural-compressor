#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
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

from .utils import quantize_elemwise_op, quantize_mx_op
from typing import Dict, Tuple
from neural_compressor.torch.utils import register_algo, set_module
from neural_compressor.common.utility import MX_QUANT
from neural_compressor.torch.quantization.config import MXQuantConfig
from neural_compressor.common.logger import Logger

logger = Logger().get_logger()
class MXLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, mx_specs=None):
        # element-wise quantize for input
        if mx_specs.get("out_dtype", "float32") != "float32":
            input = quantize_elemwise_op(input, mx_specs=mx_specs)

        if not mx_specs.get("weight_only", False):
            # MX quantize everything along input size
            input = quantize_mx_op(
                input,
                mx_specs.get("act_dtype", "bfloat16"),
                self.mx_specs["round_method"],
                self.mx_specs["blocksize"],
                axes=[-1],
            )
        # compute output
        output = F.linear(input, weight)
        if mx_specs.get("out_dtype", "float32") != "float32":
            output = quantize_elemwise_op(output, mx_specs=mx_specs)

        if bias is not None:
            output = output + bias
            if mx_specs.get("out_dtype", "float32") != "float32":
                output = quantize_elemwise_op(output, mx_specs=mx_specs)

        return output

class MXLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        mx_specs=None,
        name=None,
    ):
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = mx_specs
        super().__init__(in_features, out_features, bias)

    def apply_mx_specs(self):
        if self.mx_specs is not None:
            if self.mx_specs.get("out_dtype", "float32") != "float32":
                self.weight.data = quantize_elemwise_op(
                    self.weight.data, mx_specs=self.mx_specs
                )

                if self.bias is not None:
                    self.bias.data = quantize_elemwise_op(
                        self.bias.data, mx_specs=self.mx_specs
                    )

            # MX quantize everything along input size
            self.weight.data = quantize_mx_op(
                self.weight.data,
                self.mx_specs.get("w_dtype", "bfloat16"),
                self.mx_specs["round_method"],
                self.mx_specs["blocksize"],
                axes=[-1],
            )

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super().forward(inputs)
        
        return MXLinearFunction.apply(inputs, self.weight, self.bias, self.mx_specs)

def mx_quantize(
    model,
    config={},
    **kwargs,
):
    """Quant the model with round to nearst method.

    Args:
        model: torch module
        config (dict, optional): specific layer wise configurations. Defaults to {}.
            For example,
                config={
                    'fc2':
                        {
                            'bits': 4,
                            'group_size': 32,
                            'scheme': 'sym'
                            'gptq_perm': [1, 1, ...] # for gptq perm
                        }
                }

    Returns:
        model: fake quantized torch module
    """
    assert isinstance(model, torch.nn.Module), "only support torch module"
    supported_layers = ["Linear"]
    for name, m in model.named_modules():
        if m.__class__.__name__ not in supported_layers:
            continue
        if name not in config:  # pragma: no cover
            continue
        logger.debug(f"MX quantized module:{name, m}")
        log_msg = (
            f"MX quantization config: w_dtype={config[name]['w_dtype']}, config[name]['act_dtype'], "
            + f"out_dtype={config[name]['out_dtype']}"
        )
        logger.debug(log_msg)
        tmp_stat = m.state_dict()
        new_module = MXLinear(
            m.in_features,
            m.out_features,
            bias=m.bias is not None,
            mx_specs=config[name],
            name=name,
        )
        new_module.load_state_dict(tmp_stat)
        new_module.apply_mx_specs()
        if name == "":
            return new_module
        else:
            set_module(model, name, new_module)
    return model