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
"""Forward observer calibration for static ``TensorQuantizer`` ``amax`` statistics."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional

import torch
import torch.nn as nn

from .tensor_quantizer import TensorQuantizer


def _iter_named_quantizers(model: nn.Module) -> list[tuple[str, TensorQuantizer]]:
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, TensorQuantizer)]


def _infer_model_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        if p.device.type == "meta":
            continue
        mesh = getattr(p, "device_mesh", None)
        if mesh is not None and getattr(mesh, "device_type", None) == "cuda" and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if p.device.type != "cpu":
            return p.device
    for b in model.buffers():
        if b.device.type not in ("meta", "cpu"):
            return b.device
    return torch.device("cpu")


def _observe_static_weights(
    model: nn.Module,
    *,
    only_uncalibrated: bool = False,
    device: Optional[torch.device] = None,
) -> None:
    """Directly observe static weight quantizers so unrouted MoE experts still receive weight amax."""
    slot_pairs = (
        ("weight_quantizer", "weight", "effective_weight"),
        ("gate_up_weight_quantizer", "gate_up_proj", "effective_gate_up_weight"),
        ("gate_up_weight_quantizer", "up_proj", "effective_gate_up_weight"),
        ("down_weight_quantizer", "down_proj", "effective_down_weight"),
    )
    for mod in model.modules():
        for q_attr, w_attr, eff_fn_name in slot_pairs:
            wq = getattr(mod, q_attr, None)
            if not (isinstance(wq, TensorQuantizer) and wq.is_enabled and not wq.is_dynamic):
                continue
            if only_uncalibrated and wq.amax is not None:
                continue
            weight = getattr(mod, w_attr, None)
            if not isinstance(weight, torch.Tensor):
                continue
            is_sharded = hasattr(weight, "to_local")
            eff_fn = getattr(mod, eff_fn_name, None)
            if not is_sharded and callable(eff_fn):
                weight = eff_fn()
            wq.observe(weight)
            if device is not None and device.type != "meta" and wq.amax is not None and wq.amax.device != device:
                wq.amax = wq.amax.to(device=device, dtype=torch.float32)


def _forward_batch(model: nn.Module, batch: Any, *, device: Optional[torch.device] = None) -> None:
    target_device = device if device is not None else _infer_model_device(model)
    if isinstance(batch, Mapping):
        kwargs = {
            k: (v.to(target_device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
            if k not in ("labels", "loss_mask")
        }
        model(**kwargs)
    elif isinstance(batch, (tuple, list)):
        args = [v.to(target_device) if isinstance(v, torch.Tensor) else v for v in batch]
        model(*args)
    elif isinstance(batch, torch.Tensor):
        model(batch.to(target_device))
    else:
        raise TypeError(f"Unsupported calibration batch type {type(batch).__name__}")


def calibrate_static_amax(
    model: nn.Module,
    calib_fn: Optional[Callable[[nn.Module], Any]] = None,
    *,
    dataloader: Optional[Iterable[Any]] = None,
    num_batches: Optional[int] = None,
    distributed_sync: bool = True,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Calibrate all enabled static :class:`TensorQuantizer` modules in ``model``.

    Provide either ``calib_fn(model)`` or ``dataloader`` (optionally bounded by ``num_batches``).
    Enabled dynamic quantizers are temporarily bypassed during the calibration forward pass
    (matching ModelOpt ``enable_stats_collection``) so static activation quantizers observe clean
    reference activations, then re-enabled upon completion. Static weight quantizers are also
    observed directly so MoE experts unvisited by small calibration sets still acquire valid
    weight ``amax`` buffers.
    """
    if (calib_fn is None) == (dataloader is None):
        raise ValueError("Provide exactly one of `calib_fn` or `dataloader` to calibrate_static_amax().")

    all_quantizers = _iter_named_quantizers(model)
    named_static = [(name, q) for name, q in all_quantizers if q.is_enabled and not q.is_dynamic]
    enabled_dynamic = [q for _, q in all_quantizers if q.is_enabled and q.is_dynamic]
    was_training = model.training
    cfg_objs: list[tuple[Any, Any]] = []
    seen_cfgs: set[int] = set()
    for sub in model.modules():
        cfg = getattr(sub, "config", None)
        if cfg is not None and id(cfg) not in seen_cfgs and hasattr(cfg, "use_cache"):
            seen_cfgs.add(id(cfg))
            cfg_objs.append((cfg, getattr(cfg, "use_cache")))
            cfg.use_cache = False
    for _, q in named_static:
        q.reset_amax()
        q.enable_calib()
    for q in enabled_dynamic:
        q.disable_quant()

    resolved_device = device if device is not None else _infer_model_device(model)
    try:
        model.eval()
        with torch.no_grad():
            if calib_fn is not None:
                calib_fn(model)
            else:
                assert dataloader is not None
                for step, batch in enumerate(dataloader):
                    if num_batches is not None and step >= num_batches:
                        break
                    _forward_batch(model, batch, device=resolved_device)
            _observe_static_weights(model, only_uncalibrated=True, device=resolved_device)
    finally:
        for _, q in named_static:
            q.disable_calib()
        for q in enabled_dynamic:
            q.enable_quant()
        for cfg, prev_use_cache in cfg_objs:
            cfg.use_cache = prev_use_cache
        if was_training:
            model.train()

    if distributed_sync:
        for _, q in named_static:
            q.sync_amax_across_distributed()

    summary: dict[str, float] = {}
    for name, q in named_static:
        if q.amax is not None:
            summary[name] = float(q.amax.detach().cpu())
    return summary


def calibrate(
    model: nn.Module,
    run_fn: Callable[..., Any],
    run_args: tuple[Any, ...] = (),
    *,
    distributed: bool = True,
) -> dict[str, float]:
    """INC ``quantize(..., run_fn=..., run_args=...)`` compatible calibration entry point."""
    args = run_args if isinstance(run_args, tuple) else (run_args,)
    return calibrate_static_amax(
        model,
        calib_fn=lambda m: run_fn(m, *args),
        distributed_sync=distributed,
    )


__all__ = [
    "calibrate",
    "calibrate_static_amax",
]
