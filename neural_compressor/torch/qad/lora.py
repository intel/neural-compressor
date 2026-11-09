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
"""LoRA for QAD: standard PEFT, plus the merge-before-quant forward (path B).

Path A (``quant_then_lora``) is PEFT's default::

    y = Q(W) x + B A x

Path B (``lora_then_quant``) makes the quantizer see the merged weight::

    y = Q(W + B A) x

Path B matches a deployment that merges the adapter and then quantizes the
single model; it mirrors ModelOpt's ``_QuantLoraLinear`` merge branch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neural_compressor.torch.quantization.fake_quant import QuantLinear
from neural_compressor.torch.utils import logger

_MERGE_CLS = None


def _merge_class():
    """Build the PEFT subclass lazily so ``peft`` stays optional."""
    global _MERGE_CLS
    if _MERGE_CLS is not None:
        return _MERGE_CLS

    from peft.tuners.lora.layer import Linear as LoraLinear

    class MergeBeforeQuantLinear(LoraLinear):
        """LoRA layer whose forward merges ``B A`` before fake quantization."""

        def forward(self, x, *args, **kwargs):
            adapter_names = kwargs.pop("adapter_names", None)
            if self.disable_adapters or self.merged or adapter_names is not None:
                # Adapters off/merged: base_layer already holds W (or W+BA)
                # and fake-quantizes it itself.
                return super().forward(x, *args, **kwargs)

            base = self.base_layer
            if not isinstance(base, QuantLinear):
                raise TypeError("merge-before-quant needs QuantLinear base layers; apply quantization first")
            x = base.input_quantizer(x)

            weight = base.weight
            for adapter in self.active_adapters:
                if adapter not in self.lora_A:
                    continue
                if getattr(self, "use_dora", {}).get(adapter, False):
                    raise NotImplementedError("DoRA is not supported with merge-before-quant")
                lora_a = self.lora_A[adapter].weight
                lora_b = self.lora_B[adapter].weight
                weight = weight + ((self.scaling[adapter] * lora_b) @ lora_a).to(weight.device, weight.dtype)
            weight = base.weight_quantizer(weight)
            return F.linear(x, weight, base.bias)

    _MERGE_CLS = MergeBeforeQuantLinear
    return _MERGE_CLS


def apply_merge_before_quant(model: torch.nn.Module) -> int:
    """Rewrite every PEFT LoRA Linear in place to the path-B forward."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    merge_cls = _merge_class()
    converted = 0
    for module in model.modules():
        if isinstance(module, merge_cls):
            continue
        if not isinstance(module, LoraLinear):
            continue
        if not isinstance(module.base_layer, QuantLinear):
            raise TypeError("merge-before-quant needs QuantLinear base layers; apply quantization first")
        for dropout in module.lora_dropout.values():
            if getattr(dropout, "p", 0.0) > 0:
                raise ValueError(
                    "merge-before-quant ignores lora_dropout (the adapter is merged into the weight); "
                    "set dropout=0 or use quant_order='quant_then_lora'"
                )
        module.__class__ = merge_cls
        converted += 1
    if converted == 0:
        raise ValueError("merge-before-quant matched no PEFT LoRA Linear modules")
    logger.info(f"[lora] merge-before-quant (path B) on {converted} modules")
    return converted


def apply_lora(model: torch.nn.Module, config: dict) -> torch.nn.Module:
    """Wrap ``model`` with PEFT LoRA and optionally switch to path B.

    Args:
        model: Quantized student (``CausalLM`` or a HF model).
        config: ``{"rank": 32, "alpha": 32, "dropout": 0.0, "targets": [...],
            "quant_order": "lora_then_quant" | "quant_then_lora"}``.

    Returns:
        The ``PeftModel``. It is returned unwrapped so FSDP2 auto-wrap can see
        ``base_model.model.layers.*``.
    """
    from peft import LoraConfig, get_peft_model

    targets = list(config.get("targets") or [])
    if not targets:
        raise ValueError("LoRA config needs `targets` (module suffixes such as q_proj)")
    quant_order = config.get("quant_order", "lora_then_quant")
    if quant_order not in ("lora_then_quant", "quant_then_lora"):
        raise ValueError(f"unknown quant_order {quant_order!r}")

    lora_config = LoraConfig(
        r=config.get("rank", 32),
        lora_alpha=config.get("alpha", 32),
        lora_dropout=config.get("dropout", 0.0),
        target_modules=targets,
        bias="none",
        task_type=None,
    )
    wrapped = get_peft_model(model, lora_config, autocast_adapter_dtype=False)

    # peft creates fp32 adapters; the QAD base is bf16 and FSDP2 shards both.
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
    if dtype == torch.float32 and torch.cuda.is_available():
        dtype = torch.bfloat16
    # Cast only adapters so fp32 quantization amax buffers keep their exact values.
    for name, param in wrapped.named_parameters():
        if "lora_" in name and param.dtype != dtype:
            param.data = param.data.to(dtype)

    hits = sum(1 for name, _ in wrapped.named_modules() if name.endswith("lora_A") or name.endswith("lora_A.default"))
    if hits == 0:
        raise ValueError(f"LoRA targets {targets} matched no modules; check module names")

    if quant_order == "lora_then_quant":
        apply_merge_before_quant(wrapped)
    else:
        logger.info("[lora] adapter-after-quant (path A)")

    # Trainer calls these on the top-level model; PeftModel does not delegate.
    def _gc_enable(**kwargs):
        inner = getattr(getattr(wrapped, "base_model", None), "model", None)
        if inner is not None and hasattr(inner, "gradient_checkpointing_enable"):
            return inner.gradient_checkpointing_enable(**kwargs)

    def _gc_disable():
        inner = getattr(getattr(wrapped, "base_model", None), "model", None)
        if inner is not None and hasattr(inner, "gradient_checkpointing_disable"):
            return inner.gradient_checkpointing_disable()

    wrapped.gradient_checkpointing_enable = _gc_enable
    wrapped.gradient_checkpointing_disable = _gc_disable
    return wrapped


__all__ = ["apply_lora", "apply_merge_before_quant"]
