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
"""QAD training utilities: throughput/memory metrics, callbacks, and FSDP2 layer discovery.

All FLOPs/MFU numbers report their basis (peak table, per-device vs cluster). Unknown
devices return ``mfu=None`` instead of a made-up denominator.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import TrainerCallback

from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator

PEAK_BF16 = {
    "NVIDIA H200": 989e12,
    "NVIDIA H100": 989e12,
    "NVIDIA B200": 2250e12,
    "NVIDIA RTX PRO 6000": 1007e12,
}


def _lookup_peak_bf16(device_name: str) -> Optional[float]:
    if not device_name:
        return None
    if device_name in PEAK_BF16:
        return PEAK_BF16[device_name]
    for prefix, flops in PEAK_BF16.items():
        if device_name.startswith(prefix):
            return flops
    return None


def per_token_flops(params: int, *, mode: str = "full", teacher: bool = False, recompute: bool = False) -> float:
    """Approximate training FLOPs per token."""
    if teacher:
        return 2.0 * params
    base = 4.0 if mode == "lora" else 6.0
    return (base + (2.0 if recompute else 0.0)) * params


def tflops_and_mfu(
    *,
    tokens: int,
    step_seconds: float,
    params: int,
    mode: str = "full",
    teacher_params: int = 0,
    device_name: str = "",
    world_size: int = 1,
    recompute: bool = False,
) -> dict:
    """Return cluster and per-device rates for one optimizer step."""
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if step_seconds <= 0:
        return {"tflops": None, "mfu": None, "basis": "no-time"}
    student = tokens * per_token_flops(params, mode=mode, recompute=recompute) / step_seconds
    teacher = tokens * per_token_flops(teacher_params, teacher=True) / step_seconds
    total = student + teacher
    peak = _lookup_peak_bf16(device_name)
    aggregate_peak = peak * world_size if peak else None
    return {
        "tflops_student": student / 1e12,
        "tflops_teacher": teacher / 1e12,
        "tflops_total": total / 1e12,
        "tflops_student_per_gpu": student / world_size / 1e12,
        "tflops_teacher_per_gpu": teacher / world_size / 1e12,
        "tflops_total_per_gpu": total / world_size / 1e12,
        "mfu_student": (student / aggregate_peak) if aggregate_peak else None,
        "mfu_teacher": (teacher / aggregate_peak) if aggregate_peak else None,
        "mfu_total": (total / aggregate_peak) if aggregate_peak else None,
        "world_size": world_size,
        "recompute": recompute,
        "peak_basis": f"{device_name}-bf16-{peak / 1e12:.0f}T" if peak else "unknown",
    }


def _accelerator(device: torch.device):
    acc = getattr(torch, "accelerator", None)
    if acc is None or not acc.is_available():
        return None
    try:
        if torch.device(device).type != acc.current_accelerator().type:
            return None
    except Exception:
        return None
    return acc


def reset_peak_memory(device: torch.device | str) -> None:
    """Reset peak accelerator memory statistics on ``device`` if supported."""
    dev = torch.device(device)
    acc = _accelerator(dev)
    if acc is not None and hasattr(acc, "reset_peak_memory_stats"):
        try:
            acc.reset_peak_memory_stats(dev)
        except Exception:
            pass


def device_hardware_name(device: torch.device | str) -> str:
    """Query human-readable hardware name for ``device`` via backend module without hardcoding CUDA."""
    dev = torch.device(device)
    if dev.type == "cpu":
        return "cpu"
    backend_mod = getattr(torch, dev.type, None)
    get_name = getattr(backend_mod, "get_device_name", None) if backend_mod is not None else None
    if callable(get_name):
        try:
            return str(get_name(dev))
        except Exception:
            return ""
    return ""


def memory_report(device: torch.device | str) -> dict:
    """Return allocated/peak/reserved bytes with safe fallbacks."""
    dev = torch.device(device)
    acc = _accelerator(dev)
    if acc is None:
        return {"device": str(dev), "allocated": 0, "peak": 0, "reserved": 0}
    out = {
        "device": str(dev),
        "allocated": int(acc.memory_allocated(dev)),
        "peak": int(acc.max_memory_allocated(dev)),
    }
    for key, name in (("reserved", "memory_reserved"), ("peak_reserved", "max_memory_reserved")):
        fn = getattr(acc, name, None)
        out[key] = int(fn(dev)) if fn is not None else None
    return out


def gib(nbytes: Optional[int]) -> Optional[float]:
    """Convert bytes to GiB (``None`` passes through)."""
    return None if nbytes is None else nbytes / (1024**3)


def param_counts(model: torch.nn.Module) -> dict:
    """Total/trainable parameter counts for logging."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}


def _find_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """Walk wrapper attributes (PeftModel/VLM/DiT/DDP) looking for decoder ``nn.ModuleList``."""
    for list_attr in ("layers", "h", "blocks", "block", "transformer_blocks", "single_transformer_blocks"):
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


class GradDebugCallback(TrainerCallback):
    """Report non-finite gradients by parameter name before the optimizer step."""

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        """Scan gradients and print the offending parameters."""
        if model is None:
            return
        bad = 0
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            local = grad.to_local() if hasattr(grad, "to_local") else grad
            if not torch.isfinite(local).all():
                bad += 1
                print(
                    f"[grad-nan] step={state.global_step} {name} "
                    f"shape={tuple(local.shape)} dtype={local.dtype} "
                    f"finite={torch.isfinite(local).float().mean().item():.3f}",
                    flush=True,
                )
        if bad == 0:
            print(f"[grad-nan] step={state.global_step} all grads finite", flush=True)


class StopAfterSteps(TrainerCallback):
    """Stop after N optimizer steps while keeping a longer LR schedule."""

    def __init__(self, step: int) -> None:
        """Store the stopping step."""
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        """Request a stop once the target step is reached."""
        if self.step and state.global_step >= self.step:
            control.should_training_stop = True


__all__ = [
    "GradDebugCallback",
    "PEAK_BF16",
    "StopAfterSteps",
    "decoder_layer_class_names",
    "device_hardware_name",
    "gib",
    "memory_report",
    "param_counts",
    "per_token_flops",
    "reset_peak_memory",
    "teacher_placement",
    "tflops_and_mfu",
]
