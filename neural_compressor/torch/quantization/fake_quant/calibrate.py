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
"""Static amax calibration and lossless hex-float quantizer state persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn

from .quantizer import TensorQuantizer


def _iter_named_quantizers(model: nn.Module) -> list[tuple[str, TensorQuantizer]]:
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, TensorQuantizer)]


def _observe_static_weights(model: nn.Module, *, only_uncalibrated: bool = False) -> None:
    """Directly observe static weight quantizers so unrouted MoE experts still receive weight amax."""
    for mod in model.modules():
        wq = getattr(mod, "weight_quantizer", None)
        if not (isinstance(wq, TensorQuantizer) and wq.is_enabled and not wq.is_dynamic):
            continue
        if only_uncalibrated and wq.amax is not None:
            continue
        weight = getattr(mod, "weight", None)
        is_sharded = hasattr(weight, "to_local")
        if not is_sharded and hasattr(mod, "effective_weight") and callable(mod.effective_weight):
            weight = mod.effective_weight()
        if isinstance(weight, torch.Tensor):
            wq.observe(weight)


def _forward_batch(model: nn.Module, batch: Any) -> None:
    device = next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cpu"))
    if isinstance(batch, Mapping):
        kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
            if k not in ("labels", "loss_mask")
        }
        model(**kwargs)
    elif isinstance(batch, (tuple, list)):
        args = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
        model(*args)
    elif isinstance(batch, torch.Tensor):
        model(batch.to(device))
    else:
        raise TypeError(f"Unsupported calibration batch type {type(batch).__name__}")


def calibrate_static_amax(
    model: nn.Module,
    calib_fn: Optional[Callable[[nn.Module], Any]] = None,
    *,
    dataloader: Optional[Iterable[Any]] = None,
    num_batches: Optional[int] = None,
    distributed_sync: bool = True,
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
    named_static = [
        (name, q)
        for name, q in all_quantizers
        if q.is_enabled and not q.is_dynamic
    ]
    enabled_dynamic = [
        q
        for _, q in all_quantizers
        if q.is_enabled and q.is_dynamic
    ]
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
                    _forward_batch(model, batch)
            _observe_static_weights(model, only_uncalibrated=True)
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


def get_quantizer_state_dict(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Export calibrated static quantizer ``amax`` buffers with exact IEEE-754 ``amax_hex`` strings."""
    state: dict[str, dict[str, Any]] = {}
    for name, q in _iter_named_quantizers(model):
        if not q.is_enabled or q.is_dynamic or q.amax is None:
            continue
        val = float(q.amax.detach().to(torch.float32).cpu())
        state[name] = {
            "format": q.format.name,
            "dynamic": q.is_dynamic,
            "amax": val,
            "amax_hex": val.hex(),
        }
    return state


def _parse_state_entry(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, Mapping):
        if raw.get("amax_hex") is not None:
            return float.fromhex(str(raw["amax_hex"]))
        if raw.get("amax") is not None:
            return float(raw["amax"])
    raise ValueError(f"Invalid quantizer state entry: {raw!r}")


_LEGACY_SLOT_SUFFIXES = {
    ".weight": ".weight_quantizer",
    ".weights": ".weight_quantizer",
    ".input": ".input_quantizer",
    ".input_activations": ".input_quantizer",
    ".output": ".output_quantizer",
    ".output_activations": ".output_quantizer",
}


def _normalize_quantizer_key(key: str) -> str:
    for old_suffix, new_suffix in _LEGACY_SLOT_SUFFIXES.items():
        if key.endswith(old_suffix):
            return key[: -len(old_suffix)] + new_suffix
    return key


def _canonical_quantizer_key(key: str) -> str:
    key = _normalize_quantizer_key(key)
    changed = True
    while changed:
        changed = False
        for prefix in (
            "module.",
            "_orig_mod.",
            "_fsdp_wrapped_module.",
            "_checkpoint_wrapped_module.",
            "base_model.model.",
        ):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    for infix in (
        ".base_layer.",
        "._orig_mod.",
        "._fsdp_wrapped_module.",
        "._checkpoint_wrapped_module.",
    ):
        while infix in key:
            key = key.replace(infix, ".")
    return key


def load_quantizer_state_dict(
    model: nn.Module,
    state: Mapping[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load calibrated ``amax`` values into ``model``'s :class:`TensorQuantizer` modules."""
    raw_map = state.get("quantizers", state) if isinstance(state, Mapping) else state
    if not isinstance(raw_map, Mapping):
        raise TypeError(f"Expected mapping for quantizer state, got {type(raw_map).__name__}")

    quantizers = dict(_iter_named_quantizers(model))
    canonical_index: dict[str, tuple[str, TensorQuantizer]] = {}
    for q_name, q_mod in quantizers.items():
        canonical_index.setdefault(_canonical_quantizer_key(q_name), (q_name, q_mod))

    loaded: list[str] = []
    unexpected: list[str] = []

    for raw_key, entry in raw_map.items():
        norm_key = _normalize_quantizer_key(raw_key)
        q = quantizers.get(norm_key)
        matched_name = norm_key
        if q is None:
            canon_hit = canonical_index.get(_canonical_quantizer_key(raw_key))
            if canon_hit is not None:
                matched_name, q = canon_hit
        if q is None:
            unexpected.append(raw_key)
            continue
        q.set_amax(_parse_state_entry(entry))
        q.disable_calib()
        loaded.append(matched_name)

    loaded_set = set(loaded)
    missing = [
        name
        for name, q in quantizers.items()
        if q.is_enabled and not q.is_dynamic and q.amax is None and name not in loaded_set
    ]
    if strict and (missing or unexpected):
        parts = []
        if missing:
            parts.append(f"missing static quantizers ({len(missing)}): {missing[:5]}")
        if unexpected:
            parts.append(f"unexpected quantizer keys ({len(unexpected)}): {unexpected[:5]}")
        raise KeyError("Quantizer state mismatch: " + "; ".join(parts))

    return {"loaded": loaded, "missing": missing, "unexpected": unexpected}


def save_quantizer_state(model: nn.Module, path: Union[str, Path]) -> Path:
    """Persist calibrated quantizer state as deterministic JSON with 0-ULP ``amax_hex`` fidelity."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "quantizers": get_quantizer_state_dict(model),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


def load_quantizer_state(
    model: nn.Module,
    path: Union[str, Path],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a JSON quantizer state file written by :func:`save_quantizer_state` into ``model``."""
    payload = json.loads(Path(path).read_text())
    return load_quantizer_state_dict(model, payload, strict=strict)


__all__ = [
    "calibrate",
    "calibrate_static_amax",
    "get_quantizer_state_dict",
    "load_quantizer_state",
    "load_quantizer_state_dict",
    "save_quantizer_state",
]
