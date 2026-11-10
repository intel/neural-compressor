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
"""Optional model backends; unsloth is imported only when explicitly selected."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from .modeling import CausalLM, load_model


def wrap_unsloth(model: Any) -> CausalLM:
    """Expose an unsloth PreTrainedModel through the QAD CausalLM interface."""
    try:
        text = model.get_decoder()
    except (AttributeError, NotImplementedError):
        text = None
    try:
        lm_head = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        lm_head = None
    if text is None:
        inner = getattr(model, "model", model)
        text = getattr(inner, "language_model", inner)
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if text is None or lm_head is None:
        raise RuntimeError("unsloth backend: cannot locate text tower/lm_head")
    causal = CausalLM(text, lm_head, getattr(model, "config", getattr(text, "config", None)))
    object.__setattr__(causal, "_unsloth_source", model)
    return causal


def load_backend_model(
    backend: str,
    path: str,
    *,
    dtype: torch.dtype,
    tiny: bool = False,
    trust_remote_code: bool = False,
    max_seq_length: int = 8192,
) -> nn.Module:
    """Load a model through the selected backend with full-FT trainable defaults."""
    if backend == "hf":
        return load_model(path, dtype=dtype, tiny=tiny, trust_remote_code=trust_remote_code)
    if backend != "unsloth":
        raise ValueError(f"unknown backend {backend!r}")
    if tiny:
        raise ValueError("unsloth backend does not support the tiny CPU fixture")
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("backend='unsloth' requires the optional unsloth package") from exc

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model, _ = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
        load_in_16bit=True,
        text_only=True,
        trust_remote_code=trust_remote_code,
        device_map=f"cuda:{local_rank}",
    )
    causal = wrap_unsloth(model)
    for param in causal.parameters():
        param.requires_grad_(True)
    return causal


__all__ = ["load_backend_model", "wrap_unsloth"]
