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
"""Quantized module implementations (fake quantization)."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .options import ModuleQuantConfig, QuantizerConfig
from .quantizer import TensorQuantizer
from .registry import QuantModuleRegistry

_DISABLED = QuantizerConfig(format="bf16")


class QuantLinear(nn.Linear):
    """``nn.Linear`` with named ``weight_quantizer`` / ``input_quantizer``.

    Subclassing keeps ``weight``/``bias`` keys and ``isinstance(nn.Linear)``
    intact, which PEFT, FSDP2 and ``torch.compile`` depend on. Quantizers are
    always created (disabled when unused) so module names are stable and
    recipes that match ``*weight_quantizer`` behave predictably.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        config: Optional[ModuleQuantConfig] = None,
        **factory_kwargs: Any,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)
        config = config or ModuleQuantConfig()
        self.weight_quantizer = TensorQuantizer(config.weight or _DISABLED)
        self.input_quantizer = TensorQuantizer(config.input or _DISABLED)
        self.output_quantizer = TensorQuantizer(config.output or _DISABLED)

    @classmethod
    def from_float(
        cls,
        module: nn.Linear,
        config: ModuleQuantConfig,
        *,
        share_params: bool = False,
    ) -> "QuantLinear":
        """Build a quantized copy of ``nn.Linear``.

        ``share_params=True`` allocates the shell on the meta device and reuses
        the source ``Parameter`` objects; used when student/teacher must share
        weights without materializing a second 27B tensor.
        """
        quantized = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            config=config,
            device="meta" if share_params else module.weight.device,
            dtype=module.weight.dtype,
        )
        quantized.weight.requires_grad_(module.weight.requires_grad)
        if quantized.bias is not None and module.bias is not None:
            quantized.bias.requires_grad_(module.bias.requires_grad)
        if share_params:
            quantized.weight = module.weight
            quantized.bias = module.bias
            return quantized
        with torch.no_grad():
            quantized.weight.copy_(module.weight)
            if module.bias is not None and quantized.bias is not None:
                quantized.bias.copy_(module.bias)
        return quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize input/weight, then run the ordinary GEMM."""
        x = self.input_quantizer(x)
        # A frozen weight (LoRA, teacher) must not be saved for backward:
        # quantize it without autograd and detach. The GEMM itself stays
        # outside ``no_grad`` so gradients still reach earlier activations.
        if self.weight.requires_grad:
            weight = self.weight_quantizer(self.weight)
            out = F.linear(x, weight, self.bias)
        else:
            with torch.no_grad():
                weight = self.weight_quantizer(self.weight)
            out = F.linear(x, weight, self.bias)
        return self.output_quantizer(out)

    def extra_repr(self) -> str:
        """Append quantizer state to the default Linear repr."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_fmt={self.weight_quantizer.format.name}"
        )


QuantModuleRegistry.register(nn.Linear, QuantLinear)

__all__ = ["QuantLinear"]
