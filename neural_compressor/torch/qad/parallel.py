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

from .modeling import CausalLM, ModelAdapter, get_adapter


def _find_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """Walk wrapper attributes (PeftModel/DDP) looking for ``.layers``."""
    layers = getattr(model, "layers", None)
    if isinstance(layers, nn.ModuleList):
        return layers
    for attr in ("base", "base_model", "model", "module"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        found = _find_layers(sub)
        if found is not None:
            return found
    return None


def decoder_layer_class_names(model: nn.Module, adapter: Optional[ModelAdapter] = None) -> List[str]:
    """Class names for ``fsdp_config["transformer_layer_cls_to_wrap"]``.

    The CLI used to hardcode this list; here it comes from the actual layer
    modules, so GDN/MoE layers are covered and PeftModel wrappers are walked.
    """
    layers = _find_layers(model)
    if layers is not None and len(layers) > 0:
        return sorted({type(layer).__name__ for layer in layers})
    adapter = adapter or get_adapter(model)
    return adapter.wrap_classes(model)


def teacher_placement(model: nn.Module) -> str:
    """Return a suggested teacher placement for a loaded model.

    Small models stay on the current device; large ones (a 27B teacher next to
    a 27B student) default to FSDP2 layer sharding. ``"cpu"`` is chosen by the
    caller for the most memory-constrained runs.
    """
    if isinstance(model, CausalLM):
        return "shard"
    try:
        n_params = sum(param.numel() for param in model.parameters())
    except Exception:
        return "shard"
    return "cuda" if n_params < 8e9 else "shard"


__all__ = ["decoder_layer_class_names", "teacher_placement"]
