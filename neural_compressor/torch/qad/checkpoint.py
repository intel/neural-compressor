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
"""Checkpoint helpers for FSDP2 + PEFT.

``transformers.Trainer`` saves a ``PeftModel`` through
``accelerator.get_state_dict``. Under FSDP2 that path leaves the adapter file
empty because PEFT receives DTensors it cannot serialize. Gathering just the
adapter (a few hundred MB) avoids the problem; the optimizer state still goes
through the Trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch


def is_peft_model(model: torch.nn.Module) -> bool:
    """Return True when ``model`` is a PEFT wrapper."""
    return hasattr(model, "peft_config") and hasattr(model, "save_pretrained")


def save_full_state_dict(
    model: torch.nn.Module,
    out_dir: Union[str, Path],
    *,
    should_save: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """Gather a full-FT model from FSDP2 shards into ``model.safetensors``.

    Keys stay ``model.*`` / ``lm_head.weight``; ``export()`` maps them into the
    full VLM layout. ``cpu_offload`` keeps the gather off the GPU and only rank
    0 writes.
    """
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False),
    )
    if should_save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tensors = {key: value.detach().to(dtype=dtype).cpu().contiguous() for key, value in state.items()}
        save_file(tensors, str(out / "model.safetensors"))
    return True


def save_adapter(model: torch.nn.Module, out_dir: Union[str, Path], *, should_save: bool) -> bool:
    """Gather a PEFT adapter from FSDP2 shards and save a PEFT directory."""
    from peft import get_peft_model_state_dict
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    state = get_model_state_dict(model)
    filtered = get_peft_model_state_dict(model, state_dict=state)
    if not filtered:
        return False
    gathered = {
        key: (value.full_tensor().cpu() if hasattr(value, "full_tensor") else value.detach().cpu())
        for key, value in filtered.items()
    }
    if should_save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out, state_dict=gathered)
    return True


__all__ = ["is_peft_model", "save_adapter", "save_full_state_dict"]
