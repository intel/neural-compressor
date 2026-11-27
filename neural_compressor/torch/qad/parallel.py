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
"""Parallelism helpers: FSDP2 wrap policy and teacher placement."""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn

from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator


def _find_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """Walk wrapper attributes (PeftModel/VLM/DDP) looking for decoder ``nn.ModuleList``."""
    for list_attr in ("layers", "h", "blocks", "block"):
        layers = getattr(model, list_attr, None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    for attr in ("language_model", "base_model", "model", "transformer", "module"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module) and sub is not model:
            found = _find_layers(sub)
            if found is not None:
                return found
    return None


def decoder_layer_class_names(model: nn.Module) -> List[str]:
    """Return decoder block class names for ``fsdp_config["transformer_layer_cls_to_wrap"]``.

    Discovers actual decoder layer classes (including hybrid GatedDeltaNet + Attention or MoE
    blocks) directly from ``model`` while walking ``PeftModel`` / VLM / DDP wrappers, falling
    back to HuggingFace ``_no_split_modules`` when available.
    """
    layers = _find_layers(model)
    if layers is not None and len(layers) > 0:
        return sorted({type(layer).__name__ for layer in layers})

    cur: nn.Module = model
    for attr in ("module", "base_model", "model"):
        sub = getattr(cur, attr, None)
        if isinstance(sub, nn.Module):
            cur = sub
    no_split = getattr(cur, "_no_split_modules", None)
    if no_split:
        return sorted(str(name) for name in no_split)
    return []


def teacher_placement(model: nn.Module) -> str:
    """Return a suggested teacher placement for ``model``.

    Models under 8B parameters default to the active local accelerator (e.g. ``"cuda"`` / ``"xpu"``);
    larger models (such as a 27B teacher alongside a 27B student) default to FSDP2 ``"shard"``.
    """
    try:
        n_params = sum(param.numel() for param in model.parameters())
    except Exception:
        return "shard"
    if n_params < 8e9:
        acc_name = auto_detect_accelerator().name()
        return acc_name if acc_name in ("cuda", "cpu") else "shard"
    return "shard"


__all__ = ["decoder_layer_class_names", "teacher_placement"]
