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
"""Lossless hex-float quantizer state persistence and wrapper-prefix canonicalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Union

import torch
import torch.nn as nn

from .tensor_quantizer import TensorQuantizer

if TYPE_CHECKING:
    from .config import ModuleQuantConfig


def _iter_named_quantizers(model: nn.Module) -> list[tuple[str, TensorQuantizer]]:
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, TensorQuantizer)]


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


QUANTIZER_SLOT_NAMES: tuple[str, ...] = (
    "weight_quantizer",
    "input_quantizer",
    "output_quantizer",
    "gate_up_weight_quantizer",
    "gate_up_input_quantizer",
    "down_weight_quantizer",
    "down_input_quantizer",
)

QUANTIZER_BUFFER_MARKERS: tuple[str, ...] = tuple(f".{slot}." for slot in QUANTIZER_SLOT_NAMES)

_LEGACY_SLOT_SUFFIXES = {
    ".weight": ".weight_quantizer",
    ".weights": ".weight_quantizer",
    ".input": ".input_quantizer",
    ".input_activations": ".input_quantizer",
    ".output": ".output_quantizer",
    ".output_activations": ".output_quantizer",
}

_WRAPPER_PREFIXES = (
    "module.",
    "_orig_mod.",
    "_fsdp_wrapped_module.",
    "_checkpoint_wrapped_module.",
    "base_model.model.",
)
_WRAPPER_INFIXES = (
    ".base_layer.",
    "._orig_mod.",
    "._fsdp_wrapped_module.",
    "._checkpoint_wrapped_module.",
)
_WRAPPER_SUFFIXES = (
    ".base_layer",
    "._orig_mod",
    "._fsdp_wrapped_module",
    "._checkpoint_wrapped_module",
)


def canonicalize_module_name(name: str) -> str:
    """Strip DDP, FSDP2, activation-checkpointing, ``torch.compile``, and PEFT wrapper segments."""
    changed = True
    while changed:
        changed = False
        for prefix in _WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
        for suffix in _WRAPPER_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
    for infix in _WRAPPER_INFIXES:
        while infix in name:
            name = name.replace(infix, ".")
    return name


def _normalize_quantizer_key(key: str) -> str:
    for old_suffix, new_suffix in _LEGACY_SLOT_SUFFIXES.items():
        if key.endswith(old_suffix):
            return key[: -len(old_suffix)] + new_suffix
    return key


def canonicalize_quantizer_key(key: str) -> str:
    """Normalize a quantizer key across wrapper prefixes/infixes and legacy slot suffixes."""
    norm = _normalize_quantizer_key(key)
    for slot_name in QUANTIZER_SLOT_NAMES:
        slot_suffix = f".{slot_name}"
        if norm.endswith(slot_suffix):
            mod_part = norm[: -len(slot_suffix)]
            return f"{canonicalize_module_name(mod_part)}{slot_suffix}"
    return canonicalize_module_name(norm)


def load_amax_map_from_state(state_or_path: Union[Mapping[str, Any], str, Path]) -> dict[str, float]:
    """Load a calibrated quantizer state mapping or JSON file into ``{canonical_key: float_amax}``."""
    raw: Any = json.loads(Path(state_or_path).read_text()) if isinstance(state_or_path, (str, Path)) else state_or_path
    if isinstance(raw, Mapping) and isinstance(raw.get("quantizers"), Mapping):
        raw = raw["quantizers"]
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected quantizer state mapping, got {type(raw).__name__}")
    resolved: dict[str, float] = {}
    for key, val in raw.items():
        if val is None:
            continue
        amax_val = _parse_state_entry(val)
        norm_key = _normalize_quantizer_key(str(key))
        canon_key = canonicalize_quantizer_key(str(key))
        resolved[norm_key] = amax_val
        resolved.setdefault(canon_key, amax_val)
    return resolved


def apply_amax_map_to_module_config(
    module_name: str,
    module_cfg: "ModuleQuantConfig",
    amax_map: Mapping[str, float],
) -> None:
    """Populate static ``amax`` fields on ``module_cfg`` from a pre-calibrated state map."""
    canonical = canonicalize_module_name(module_name)
    prefixes = (module_name,) if canonical == module_name else (module_name, canonical)
    slot_candidates = (
        (
            module_cfg.weights,
            ("weight_quantizer", "weight", "weights", "gate_up_weight_quantizer", "down_weight_quantizer"),
        ),
        (
            module_cfg.input_activations,
            ("input_quantizer", "input", "input_activations", "gate_up_input_quantizer", "down_input_quantizer"),
        ),
        (module_cfg.output_activations, ("output_quantizer", "output", "output_activations")),
    )
    for q_cfg, suffixes in slot_candidates:
        if q_cfg.is_noop:
            continue
        found = False
        for prefix in prefixes:
            for suffix in suffixes:
                candidate = f"{prefix}.{suffix}"
                if candidate in amax_map:
                    q_cfg.amax = float(amax_map[candidate])
                    found = True
                    break
            if found:
                break


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
        canonical_index.setdefault(canonicalize_quantizer_key(q_name), (q_name, q_mod))

    loaded: list[str] = []
    unexpected: list[str] = []

    for raw_key, entry in raw_map.items():
        norm_key = _normalize_quantizer_key(raw_key)
        q = quantizers.get(norm_key)
        matched_name = norm_key
        if q is None:
            canon_hit = canonical_index.get(canonicalize_quantizer_key(raw_key))
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
    "apply_amax_map_to_module_config",
    "canonicalize_module_name",
    "canonicalize_quantizer_key",
    "get_quantizer_state_dict",
    "load_amax_map_from_state",
    "load_quantizer_state",
    "load_quantizer_state_dict",
    "save_quantizer_state",
]
