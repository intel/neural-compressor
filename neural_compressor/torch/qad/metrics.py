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
"""Training metrics: FLOPs/MFU, memory and parameter counts.

All numbers report their basis (peak table, per-device vs cluster). Unknown
devices return ``mfu=None`` instead of a made-up denominator.
"""

from __future__ import annotations

from typing import Optional

import torch

PEAK_BF16 = {
    "NVIDIA H200": 989e12,
    "NVIDIA H100": 989e12,
    "NVIDIA B200": 2250e12,
}


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
    peak = PEAK_BF16.get(device_name)
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


__all__ = ["PEAK_BF16", "gib", "memory_report", "param_counts", "per_token_flops", "tflops_and_mfu"]
