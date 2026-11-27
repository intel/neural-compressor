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
"""Fake-quantized module subclasses (``QuantLinear``)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModuleQuantConfig
from .quantizer import TensorQuantizer
from .registry import register_quant_module


@register_quant_module(nn.Linear)
class QuantLinear(nn.Linear):
    """Drop-in ``nn.Linear`` subclass with weight, input, and output :class:`TensorQuantizer` slots.

    Shares ``weight`` and ``bias`` ``nn.Parameter`` objects in-place via :meth:`from_float`
    without duplicating memory or altering ``state_dict`` parameter keys.
    """

    _is_fake_quant_module: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: ModuleQuantConfig | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        resolved = config if config is not None else ModuleQuantConfig()
        self.quant_config = resolved
        self.weight_quantizer = TensorQuantizer(resolved.weights)
        self.input_quantizer = TensorQuantizer(resolved.input_activations)
        self.output_quantizer = TensorQuantizer(resolved.output_activations)

    @classmethod
    def from_float(cls, mod: nn.Linear, config: ModuleQuantConfig) -> "QuantLinear":
        """Construct a :class:`QuantLinear` sharing ``mod.weight`` and ``mod.bias`` without copying."""
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"QuantLinear.from_float expects nn.Linear, got {type(mod).__name__}")
        # Allocate shell on meta device so zero memory is allocated before sharing parameters.
        qlinear = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            config=config,
            device="meta",
            dtype=mod.weight.dtype,
        )
        qlinear.weight = mod.weight
        qlinear.bias = mod.bias
        param_device = mod.weight.device
        if param_device.type != "meta":
            for q_mod in (qlinear.weight_quantizer, qlinear.input_quantizer, qlinear.output_quantizer):
                if q_mod.amax is not None:
                    q_mod.amax = q_mod.amax.to(device=param_device, dtype=torch.float32)
        return qlinear

    def to_float(self) -> nn.Linear:
        """Unwrap back into a plain ``nn.Linear`` sharing ``weight`` and ``bias``."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device="meta",
            dtype=self.weight.dtype,
        )
        linear.weight = self.weight
        linear.bias = self.bias
        return linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_input = self.input_quantizer(input)
        q_weight = self.weight_quantizer(self.weight)
        if q_weight.dtype != q_input.dtype:
            q_weight = q_weight.to(q_input.dtype)
        bias = self.bias
        if bias is not None and bias.dtype != q_input.dtype:
            bias = bias.to(q_input.dtype)
        output = F.linear(q_input, q_weight, bias)
        return self.output_quantizer(output)


__all__ = ["QuantLinear"]
