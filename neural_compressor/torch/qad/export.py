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
"""Export fake-quant-trained checkpoints to HF format.

v1 writes **bf16** weights (fake-quant training keeps float parameters). The
artifact is a normal HF transformers model: text tower from the trained
weights, vision tower/processor from the base checkpoint. Real MXFP4/FP8
export (reusing the calibrated amax) is a separate follow-up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import torch

from neural_compressor.torch.quantization.fake_quant import FakeQuantConfig, TensorQuantizer
from neural_compressor.torch.utils import logger

from .config import ExportConfig

QUANT_SCHEMA = "neural_compressor.qad/v1"


def _buffer_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "full_tensor"):
        value = value.full_tensor()
    return float(value.detach().reshape(()))


def write_quantization_config(model: torch.nn.Module, out_dir: str | Path, quant_config: FakeQuantConfig) -> Path:
    """Write ``quantization_config.json`` with the plan and static amax values."""
    amax = {}
    for name, module in model.named_modules():
        for field in ("weight", "input", "output"):
            quantizer = getattr(module, f"{field}_quantizer", None)
            if not isinstance(quantizer, TensorQuantizer):
                continue
            if quantizer.scheme != "static" or quantizer.amax is None:
                continue
            amax[f"{name}.{field}"] = _buffer_to_float(quantizer.amax)
    payload = {
        "schema": QUANT_SCHEMA,
        "mode": "fake_quant",
        "quant": quant_config.to_dict() if quant_config is not None else None,
        "amax": amax,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "quantization_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"[export] wrote {path} ({len(amax)} static amax values)")
    return path


def _gather_state_dict(model: torch.nn.Module, dtype: torch.dtype) -> dict:
    """Materialize a (possibly FSDP2-sharded) model as CPU tensors."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    try:
        state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False),
        )
    except Exception as exc:  # not distributed / not sharded
        logger.debug(f"[export] falling back to state_dict(): {exc}")
        state = model.state_dict()
    return {key: value.detach().to(dtype=dtype).cpu().contiguous() for key, value in state.items()}


def load_state_dict_into_model(model: torch.nn.Module, path: str | Path) -> int:
    """Load a ``model.safetensors`` produced by the QAD trainer into ``model``."""
    from safetensors import safe_open

    state = {}
    with safe_open(str(path), "pt") as handle:
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"{len(missing)} tensors missing from {path}: {missing[:5]}")
    if unexpected:
        raise ValueError(f"{len(unexpected)} unexpected tensors in {path}: {unexpected[:5]}")
    logger.info(f"[export] loaded {len(state)} tensors from {path}")
    return len(state)


def _merge_lora(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    inner = getattr(model, "peft_model", None)
    if inner is not None and hasattr(inner, "merge_and_unload"):
        return inner.merge_and_unload()
    return model


def export(
    model: torch.nn.Module,
    tokenizer: Any,
    config: ExportConfig,
    out_dir: str | Path,
    *,
    base_model: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Export a trained (optionally LoRA) model to a HF directory.

    Args:
        model: Trained model (``CausalLM`` wrapper or a HF model).
        tokenizer: Tokenizer/processor to save next to the weights.
        config: Export format/layout (v1 only ``hf_bf16``).
        out_dir: Destination directory.
        base_model: Base checkpoint used to restore the vision tower and
            processor. When ``None`` and ``config.keep_vision`` is True the
            export falls back to text-only.
        dtype: Export dtype (bf16 by default).
    """
    if config.format != "hf_bf16":
        raise ValueError(f"unsupported export format {config.format!r}")
    model = _merge_lora(model)
    state = _gather_state_dict(model, dtype)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if config.keep_vision and base_model:
        _export_full_vlm(state, tokenizer, base_model, out, dtype)
    else:
        _export_text(state, tokenizer, model, out)
    logger.info(f"[export] wrote {out} ({len(state)} tensors, dtype={dtype})")
    return out


def _export_text(state: dict, tokenizer: Any, model: torch.nn.Module, out: Path) -> None:
    from safetensors.torch import save_file

    save_file(state, str(out / "model.safetensors"))
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "to_dict"):
        payload = cfg.to_dict()
        architectures = list(payload.get("architectures") or [])
        if (not architectures or architectures[0].endswith("TextModel")) and payload.get(
            "model_type"
        ) == "qwen3_5_text":
            architectures = ["Qwen3_5ForCausalLM"]
        payload["architectures"] = architectures
        (out / "config.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if tokenizer is not None:
        tokenizer.save_pretrained(out)


def _export_full_vlm(state: dict, tokenizer: Any, base_model: str, out: Path, dtype: torch.dtype) -> None:
    """Restore the base VLM, overwrite its text tower, and save the whole model."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    full = AutoModelForImageTextToText.from_pretrained(
        base_model, dtype=dtype, low_cpu_mem_usage=True, device_map="cpu"
    )
    _load_text_state_dict(full, state)
    full.save_pretrained(out, safe_serialization=True, max_shard_size="10GB")
    try:
        AutoProcessor.from_pretrained(base_model).save_pretrained(out)
    except Exception:
        AutoTokenizer.from_pretrained(base_model).save_pretrained(out)
    (out / "export_info.json").write_text(
        json.dumps({"backend": "hf-vlm", "method": "merged_16bit", "base": base_model}, indent=2),
        encoding="utf-8",
    )


def _load_text_state_dict(full: torch.nn.Module, state: dict) -> int:
    """Copy text-tower tensors (``model.*`` / ``lm_head.weight``) into a VLM.

    The VLM stores the tower under ``model.language_model.*``; vision weights
    are untouched. Raises when the counts do not match, so a partial export
    never goes unnoticed.
    """
    expected = sum(
        1 for name, _ in full.named_parameters() if name.startswith("model.language_model.") or name == "lm_head.weight"
    )
    loaded = 0
    for key, value in state.items():
        if key == "lm_head.weight":
            target = full.lm_head.weight
        elif key.startswith("model."):
            target = full.get_parameter("model.language_model." + key[len("model.") :])
        else:
            raise KeyError(f"unexpected text state-dict key: {key}")
        target.data.copy_(value.to(target.dtype))
        loaded += 1
    if loaded != expected:
        raise ValueError(f"loaded {loaded} tensors, expected {expected}")
    return loaded


__all__ = ["QUANT_SCHEMA", "export", "load_state_dict_into_model", "write_quantization_config"]
