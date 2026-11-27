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
"""PEFT LoRA fake-quantization plugin supporting Path-B (merge-before-quant) and Path-A."""

from __future__ import annotations

import importlib.util
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModuleQuantConfig
from ..modules import QuantLinear
from ..quantizer import TensorQuantizer
from ..registry import register_quant_module

QuantOrder = Literal["lora_then_quant", "quant_then_lora"]

_PEFT_REGISTERED: bool = False


def is_peft_available() -> bool:
    """Return True if ``peft`` is importable in the active Python environment."""
    return importlib.util.find_spec("peft") is not None


def ensure_peft_torchao_compat() -> None:
    """Guard against PEFT raising ``ImportError`` when an older ``torchao`` (<0.16) is installed."""
    if not is_peft_available():
        return
    try:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_torchao_available()
    except ImportError:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_torchao_available = lambda: False
        try:
            import peft.tuners.lora.torchao as peft_lora_torchao

            peft_lora_torchao.is_torchao_available = lambda: False
        except Exception:
            pass


if is_peft_available():
    ensure_peft_torchao_compat()
    from peft.tuners.lora.layer import Linear as LoraLinear
else:  # pragma: no cover - fallback stub when peft is absent
    LoraLinear = nn.Module  # type: ignore[assignment,misc]


def _require_peft() -> None:
    if not is_peft_available():
        raise ImportError("PEFT LoRA quantization support requires the `peft` package (`pip install peft`).")
    ensure_peft_torchao_compat()


class QuantLoraLinear(LoraLinear):
    """Fake-quantized PEFT ``LoraLinear`` supporting both deployment-aligned Path-B and Path-A.

    * ``quant_order="lora_then_quant"`` (Path B, default):
      ``y = Q_out(F.linear(Q_in(x), Q_w(W_base + sum_i ((s_i * B_i) @ A_i)), b))``
      Matches NVIDIA ``ModelOpt``'s ``_QuantLoraLinear`` and ``Megatron-Bridge``'s ``MergedLoRALinear``
      bit-for-bit, ensuring zero train-to-export quantization mismatch once adapters are merged.
    * ``quant_order="quant_then_lora"`` (Path A):
      ``y = Q_out(F.linear(Q_in(x), Q_w(W_base), b)) + sum_i s_i * B_i(A_i(dropout_i(x)))``
      Matches standard PEFT wrapping a ``QuantLinear`` base layer.
    """

    _is_fake_quant_module: bool = True

    @classmethod
    def from_lora(
        cls,
        mod: nn.Module,
        config: Optional[ModuleQuantConfig] = None,
        *,
        quant_order: QuantOrder = "lora_then_quant",
    ) -> "QuantLoraLinear":
        """Convert an existing ``peft.tuners.lora.Linear`` into :class:`QuantLoraLinear`."""
        _require_peft()
        if not isinstance(mod, LoraLinear):
            raise TypeError(f"QuantLoraLinear.from_lora expects peft.tuners.lora.Linear, got {type(mod).__name__}")
        if quant_order not in ("lora_then_quant", "quant_then_lora"):
            raise ValueError(
                f"Unknown quant_order {quant_order!r}; expected 'lora_then_quant' or 'quant_then_lora'."
            )
        if quant_order == "lora_then_quant":
            for dropout in mod.lora_dropout.values():
                if getattr(dropout, "p", 0.0) > 0:
                    raise ValueError(
                        "quant_order='lora_then_quant' merges LoRA weights prior to quantization and cannot apply "
                        "activation-space lora_dropout > 0; set lora_dropout=0.0 or use quant_order='quant_then_lora'."
                    )

        raw_base = mod.base_layer
        inherited_quantizers: Optional[tuple[TensorQuantizer, TensorQuantizer, TensorQuantizer]] = None
        inherited_cfg: Optional[ModuleQuantConfig] = None
        if isinstance(raw_base, QuantLinear):
            inherited_quantizers = (
                raw_base.weight_quantizer,
                raw_base.input_quantizer,
                raw_base.output_quantizer,
            )
            inherited_cfg = raw_base.quant_config
            base_linear = raw_base.to_float()
        elif isinstance(raw_base, nn.Linear):
            base_linear = raw_base
        else:
            raise TypeError(
                f"QuantLoraLinear requires an nn.Linear or QuantLinear base_layer, got {type(raw_base).__name__}"
            )

        qlora = cls.__new__(cls)
        qlora.__dict__.update(mod.__dict__)
        qlora._modules = dict(mod._modules)
        qlora._parameters = dict(mod._parameters)
        qlora._buffers = dict(mod._buffers)
        qlora._non_persistent_buffers_set = set(mod._non_persistent_buffers_set)
        qlora.base_layer = base_linear
        qlora.quant_order = quant_order

        if config is not None:
            qlora.quant_config = config
            qlora.weight_quantizer = TensorQuantizer(config.weights)
            qlora.input_quantizer = TensorQuantizer(config.input_activations)
            qlora.output_quantizer = TensorQuantizer(config.output_activations)
        elif inherited_quantizers is not None:
            qlora.quant_config = inherited_cfg if inherited_cfg is not None else ModuleQuantConfig()
            qlora.weight_quantizer, qlora.input_quantizer, qlora.output_quantizer = inherited_quantizers
        else:
            raise ValueError(
                "QuantLoraLinear.from_lora requires `config` when `mod.base_layer` is not already a QuantLinear."
            )

        param_device = base_linear.weight.device
        if param_device.type != "meta":
            for q_mod in (qlora.weight_quantizer, qlora.input_quantizer, qlora.output_quantizer):
                if q_mod.amax is not None:
                    q_mod.amax = q_mod.amax.to(device=param_device, dtype=torch.float32)
        return qlora

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """Compute ``((scaling * B) @ A).to(W.device, W.dtype)`` matching ModelOpt's Path-B association."""
        if self.quant_order != "lora_then_quant":
            return super().get_delta_weight(adapter)
        base_weight = self.base_layer.weight
        weight_a = self.lora_A[adapter].weight
        weight_b = self.lora_B[adapter].weight
        scaling = self.scaling[adapter]
        delta = (scaling * weight_b) @ weight_a
        if getattr(self, "fan_in_fan_out", False):
            delta = delta.T
        return delta.to(device=base_weight.device, dtype=base_weight.dtype)

    def effective_weight(self) -> torch.Tensor:
        """Return ``W_base + sum_{active} delta_W`` in ``W_base``'s dtype and device."""
        weight = self.base_layer.weight
        if self.disable_adapters or self.merged:
            return weight
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            if getattr(self, "use_dora", {}).get(active_adapter, False):
                raise NotImplementedError("DoRA is not supported in QuantLoraLinear.")
            lora_a = self.lora_A[active_adapter].weight
            lora_b = self.lora_B[active_adapter].weight
            scaling = self.scaling[active_adapter]
            delta = (scaling * lora_b) @ lora_a
            if getattr(self, "fan_in_fan_out", False):
                delta = delta.T
            weight = weight + delta.to(device=weight.device, dtype=weight.dtype)
        return weight

    def _base_quant_linear_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        q_input = self.input_quantizer(x)
        q_weight = self.weight_quantizer(weight)
        if q_weight.dtype != q_input.dtype:
            q_weight = q_weight.to(q_input.dtype)
        bias = self.base_layer.bias
        if bias is not None and bias.dtype != q_input.dtype:
            bias = bias.to(q_input.dtype)
        out = F.linear(q_input, q_weight, bias)
        return self.output_quantizer(out)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is not None:
            return super().forward(x, *args, adapter_names=adapter_names, **kwargs)

        if self.disable_adapters or self.merged:
            return self._base_quant_linear_forward(x, self.base_layer.weight)

        if self.quant_order == "lora_then_quant":
            return self._base_quant_linear_forward(x, self.effective_weight())

        # Path A ("quant_then_lora"): quantized base linear + unquantized high-precision LoRA branch.
        result = self._base_quant_linear_forward(x, self.base_layer.weight)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            if getattr(self, "use_dora", {}).get(active_adapter, False):
                raise NotImplementedError("DoRA is not supported in QuantLoraLinear.")
            lora_a = self.lora_A[active_adapter]
            lora_b = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x_cast = self._cast_input_dtype(x, lora_a.weight.dtype)
            result = result + lora_b(lora_a(dropout(x_cast))) * scaling
        return result.to(torch_result_dtype)

    def to_quant_linear(self, *, merge_weights: bool = True) -> QuantLinear:
        """Collapse this LoRA wrapper into a standalone :class:`QuantLinear` preserving quantizer states."""
        with torch.no_grad():
            if merge_weights and not self.merged:
                if self.quant_order == "lora_then_quant":
                    merged_weight = self.effective_weight().detach().clone()
                else:
                    merged_weight = self.base_layer.weight.detach().clone()
                    for active_adapter in self.active_adapters:
                        if active_adapter in self.lora_A:
                            merged_weight = merged_weight + self.get_delta_weight(active_adapter).to(
                                device=merged_weight.device, dtype=merged_weight.dtype
                            )
            else:
                merged_weight = self.base_layer.weight.detach().clone()

            merged_bias = self.base_layer.bias.detach().clone() if self.base_layer.bias is not None else None

        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=merged_bias is not None,
            device="meta",
            dtype=merged_weight.dtype,
        )
        linear.weight = nn.Parameter(merged_weight, requires_grad=self.base_layer.weight.requires_grad)
        linear.bias = (
            nn.Parameter(
                merged_bias,
                requires_grad=self.base_layer.bias.requires_grad if self.base_layer.bias is not None else False,
            )
            if merged_bias is not None
            else None
        )

        qlinear = QuantLinear.from_float(linear, self.quant_config)
        qlinear.weight_quantizer = self.weight_quantizer
        qlinear.input_quantizer = self.input_quantizer
        qlinear.output_quantizer = self.output_quantizer
        return qlinear


def register_peft_quant_modules() -> bool:
    """Register ``peft.tuners.lora.Linear -> QuantLoraLinear.from_lora`` in ``fake_quant``'s registry."""
    global _PEFT_REGISTERED
    if _PEFT_REGISTERED:
        return True
    if not is_peft_available():
        return False
    ensure_peft_torchao_compat()
    register_quant_module(LoraLinear, QuantLoraLinear.from_lora)
    _PEFT_REGISTERED = True
    return True


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    if "." in dotted_name:
        parent_name, child_name = dotted_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = dotted_name
    setattr(parent, child_name, new_module)


def convert_peft_lora_to_quant_lora(
    model: nn.Module,
    *,
    quant_order: QuantOrder = "lora_then_quant",
) -> int:
    """Upgrade ``LoraLinear(base_layer=QuantLinear)`` modules in ``model`` into :class:`QuantLoraLinear`."""
    _require_peft()
    targets: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLoraLinear):
            mod.quant_order = quant_order
            continue
        if isinstance(mod, LoraLinear) and isinstance(mod.base_layer, QuantLinear):
            targets.append((name, mod))

    for name, mod in targets:
        _set_submodule(model, name, QuantLoraLinear.from_lora(mod, quant_order=quant_order))
    return len(targets)


def merge_lora_into_quant_linear(model: nn.Module) -> nn.Module:
    """Merge all PEFT LoRA adapters in ``model`` into :class:`QuantLinear` (or ``nn.Linear``) and unwrap ``PeftModel``."""
    if not is_peft_available():
        return model
    ensure_peft_torchao_compat()

    replacements: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if not name:
            continue
        if isinstance(mod, QuantLoraLinear):
            replacements.append((name, mod.to_quant_linear(merge_weights=True)))
        elif isinstance(mod, LoraLinear):
            mod.merge()
            replacements.append((name, mod.get_base_layer()))

    for name, new_mod in replacements:
        _set_submodule(model, name, new_mod)

    quant_cfg = getattr(model, "quant_config", None)
    if hasattr(model, "peft_config") and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        unwrapped = model.base_model.model
        if quant_cfg is not None and getattr(unwrapped, "quant_config", None) is None:
            unwrapped.quant_config = quant_cfg
        return unwrapped
    return model


__all__ = [
    "QuantLoraLinear",
    "QuantOrder",
    "convert_peft_lora_to_quant_lora",
    "ensure_peft_torchao_compat",
    "is_peft_available",
    "merge_lora_into_quant_linear",
    "register_peft_quant_modules",
]
