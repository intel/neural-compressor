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
"""MX quantization."""

from collections import OrderedDict

import torch
from torch.nn import functional as F

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.utils import logger, set_module

from .utils import quantize_elemwise_op, quantize_mx_op


class MXLinear(torch.nn.Linear):
    """Linear for MX data type."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        mx_specs=None,
        name=None,
    ):
        """Initialization function."""
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = mx_specs
        super().__init__(in_features, out_features, bias)

    def apply_mx_specs(self):
        """Apply MX data type to weight."""
        if self.mx_specs is not None:
            if self.mx_specs.out_dtype != "float32":
                self.weight.data = quantize_elemwise_op(self.weight.data, mx_specs=self.mx_specs)

                if self.bias is not None:
                    self.bias.data = quantize_elemwise_op(self.bias.data, mx_specs=self.mx_specs)

            # MX quantize everything along input size
            self.weight.data = quantize_mx_op(
                self.weight.data,
                self.mx_specs.w_dtype,
                self.mx_specs.round_method,
                self.mx_specs.blocksize,
                axes=[-1],
            )

    def forward(self, input):
        """Forward function."""
        if self.mx_none:
            return super().forward(input)

        # element-wise quantize for input
        if self.mx_specs.out_dtype != "float32":
            input = quantize_elemwise_op(input, mx_specs=self.mx_specs)

        if not self.mx_specs.weight_only:
            # MX quantize everything along input size
            input = quantize_mx_op(
                input,
                self.mx_specs.act_dtype,
                self.mx_specs.round_method,
                self.mx_specs.blocksize,
                axes=[-1],
            )
        # compute output
        output = F.linear(input, self.weight)
        if self.mx_specs.out_dtype != "float32":
            output = quantize_elemwise_op(output, mx_specs=self.mx_specs)

        if self.bias is not None:
            output = output + self.bias
            if self.mx_specs.out_dtype != "float32":
                output = quantize_elemwise_op(output, mx_specs=self.mx_specs)

        return output


class MXQuantizer(Quantizer):
    """Quantizer of MX data type."""

    def __init__(self, quant_config: OrderedDict = {}):
        """Init a MXQuantizer object.

        Args:
            quant_config (OrderedDict, optional): quantization config for ops. Defaults to {}.
        """
        super().__init__(quant_config)

    @torch.no_grad()
    def prepare(self, model, *args, **kwargs):
        """Prepares a given model for quantization.

        Will return model directly in MX quant algorithm.

        Args:
            model (torch.nn.Module): The model to be prepared.
        """
        return model

    @torch.no_grad()
    def convert(
        self,
        model,
        **kwargs,
    ):
        """Quant the model with mx data type.

        Args:
            model: torch module

        Returns:
            model: fake quantized torch module
        """
        assert isinstance(model, torch.nn.Module), "only support torch module"
        supported_layers = ["Linear"]
        config = self.quant_config
        for name, m in model.named_modules():
            if type(m).__name__ not in supported_layers:
                continue
            if (name, type(m).__name__) not in config:  # pragma: no cover
                continue
            logger.debug(f"MX quantized module:{name, m}")
            log_msg = (
                f"MX quantization config: w_dtype={config[(name, type(m).__name__)].w_dtype}, "
                + "config[(name, type(m).__name__)].act_dtype, "
                + f"out_dtype={config[(name, type(m).__name__)].out_dtype}"
            )
            logger.debug(log_msg)
            tmp_stat = m.state_dict()
            new_module = MXLinear(
                m.in_features,
                m.out_features,
                bias=m.bias is not None,
                mx_specs=config[(name, type(m).__name__)],
                name=name,
            )
            new_module.load_state_dict(tmp_stat)
            new_module.apply_mx_specs()
            if name == "":
                return new_module
            else:
                set_module(model, name, new_module)
        return model
