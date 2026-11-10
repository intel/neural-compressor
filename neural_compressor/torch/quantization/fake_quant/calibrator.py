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
"""Calibration: where the static amax values come from.

Three sources, in priority order:

1. ``.quant_summary.txt`` written by ModelOpt PTQ calibration -- exact fp32
   ``amax`` as text. This is what the 27B recipe uses.
2. A quantized HF export's ``weight_scale`` / ``input_scale`` tensors --
   ``amax / 448`` after a bf16 round-trip (~0.1% off, can move 22/256 modules
   by 1 ulp, which accumulates over 64 FP8 layers).
3. A forward loop over calibration data (``run_fn``), i.e. MaxCalibrator.

Only static per-tensor FP8 quantizers consume these values; MXFP4/NVFP4 block
formats compute their scales dynamically and have ``amax=None``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import torch

from neural_compressor.torch.utils import logger

from .quantizer import TensorQuantizer

_AMAX_RE = re.compile(r"amax=([0-9.eE+-]+)")
_MEGATRON_RE = re.compile(
    r"(?:language_model\.)?decoder\.layers\.(\d+)\.self_attention\.([a-z_]+)\.(input|weight)_quantizer"
)

# Megatron module name -> HF suffixes (relative to the decoder layer list).
_MEGATRON_SUFFIXES = {
    "linear_qkv": [f"self_attn.{name}_proj" for name in ("q", "k", "v")],
    "linear_proj": ["self_attn.o_proj"],
    "in_proj": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
    "out_proj": ["linear_attn.out_proj"],
}


def modelopt_megatron_translate(name: str) -> list[str]:
    """Translate a ModelOpt/Megatron quantizer name into HF module suffixes.

    ``language_model.decoder.layers.0.self_attention.linear_qkv.input_quantizer``
    becomes ``["layers.0.self_attn.q_proj", ...]``; the caller matches these as
    suffixes so both ``model.layers.*`` and ``model.language_model.layers.*``
    layouts work.
    """
    match = _MEGATRON_RE.match(name)
    if match is None:
        return []
    layer, module = int(match.group(1)), match.group(2)
    suffixes = _MEGATRON_SUFFIXES.get(module)
    if suffixes is None:
        return []
    return [f"layers.{layer}.{suffix}" for suffix in suffixes]


def parse_quant_summary(path: str | Path) -> Dict[str, Dict[str, torch.Tensor]]:
    """Parse ModelOpt's ``.quant_summary.txt`` into ``{name: {field: amax}}``."""
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        amax_match = _AMAX_RE.search(line)
        if amax_match is None:
            continue
        name = parts[0]
        field_name = "weight" if name.endswith("weight_quantizer") else "input"
        out.setdefault(name, {})[field_name] = torch.tensor(float(amax_match.group(1)))
    return out


def load_amax(
    path: str | Path,
    *,
    translate: Optional[Callable[[str], Iterable[str]]] = modelopt_megatron_translate,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load an amax map from a summary txt or a ``quantization_config.json``.

    Args:
        path: ``.txt`` (ModelOpt summary) or ``.json`` (our export metadata).
        translate: Maps a summary quantizer name to HF module suffixes. Ignored
            for JSON, whose keys are already module names.

    Returns:
        ``{module_suffix_or_name: {"weight"|"input": amax}}``.
    """
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        raw = payload.get("amax", payload)
        result = {}
        for name, value in raw.items():
            if isinstance(value, dict):
                fields = value
            else:
                name, field = name.rsplit(".", 1)
                fields = {field: value}
            for field, item in fields.items():
                result.setdefault(name, {})[field] = torch.tensor(float(item))
        return result
    if path.suffix not in (".txt", ""):
        raise ValueError(f"unsupported amax file {path}; expected .txt or .json")

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for quantizer_name, fields in parse_quant_summary(path).items():
        suffixes = list(translate(quantizer_name)) if translate is not None else []
        if not suffixes:
            continue
        for suffix in suffixes:
            out.setdefault(suffix, {}).update(fields)
    if not out:
        raise ValueError(f"{path}: no amax entries could be mapped; check --translate / names")
    return out


class AmaxMap:
    """Look up amax by exact module name or unique suffix match."""

    def __init__(self, mapping: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Store the mapping produced by :func:`load_amax`."""
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.mapping)

    def lookup(self, module_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """Return ``{"weight"/"input": amax}`` for a module, if present."""
        if module_name in self.mapping:
            return self.mapping[module_name]
        matches = [
            fields
            for suffix, fields in self.mapping.items()
            if module_name.endswith("." + suffix) or module_name == suffix
        ]
        if len(matches) > 1:
            raise ValueError(f"ambiguous amax suffix match for {module_name!r}")
        return matches[0] if matches else None


def apply_amax(
    model: torch.nn.Module,
    plan: Dict[str, object],
    amax: AmaxMap,
    *,
    strict: bool = True,
) -> dict:
    """Assign calibrated amax to every static quantizer in the plan.

    Raises when a static quantizer has no value and ``strict`` is True: silently
    falling back to dynamic FP8 changes the numerics of every FP8 layer.
    """
    assigned, missing = 0, []
    for name, config in plan.items():
        module = model.get_submodule(name)
        for field_name in ("weight", "input", "output"):
            quantizer_cfg = getattr(config, field_name)
            if quantizer_cfg is None or quantizer_cfg.scheme != "static":
                continue
            quantizer = getattr(module, f"{field_name}_quantizer")
            if not isinstance(quantizer, TensorQuantizer):
                continue
            fields = amax.lookup(name) or {}
            value = fields.get(field_name)
            if value is None:
                missing.append(f"{name}.{field_name}")
                continue
            quantizer.load_amax(value)
            assigned += 1
    if missing and strict:
        head = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} static quantizers have no amax from the given source: {head} ... "
            "pass a complete summary/JSON or run calibration"
        )
    logger.info(f"[calib] assigned {assigned} amax values from file, missing={len(missing)}")
    return {"assigned": assigned, "missing": missing}


def calibrate(
    model: torch.nn.Module,
    run_fn: Callable,
    run_args: Optional[Iterable] = None,
    run_kwargs: Optional[dict] = None,
) -> None:
    """Run a calibration forward loop and finalize every static quantizer.

    ``run_fn`` receives ``(model, *run_args, **run_kwargs)`` and is expected to
    forward calibration batches through the model.
    """
    quantizers = [m for m in model.modules() if isinstance(m, TensorQuantizer)]
    for quantizer in quantizers:
        if quantizer.scheme == "static" and quantizer.enabled:
            quantizer.calibrating = True
    try:
        run_fn(model, *(run_args or ()), **(run_kwargs or {}))
    finally:
        for quantizer in quantizers:
            if quantizer.scheme == "static":
                quantizer.calibrating = False
    for quantizer in quantizers:
        quantizer.finalize()
    logger.info(f"[calib] calibrated {sum(1 for q in quantizers if q.enabled)} quantizers via run_fn")


__all__ = [
    "AmaxMap",
    "apply_amax",
    "calibrate",
    "load_amax",
    "modelopt_megatron_translate",
    "parse_quant_summary",
]
