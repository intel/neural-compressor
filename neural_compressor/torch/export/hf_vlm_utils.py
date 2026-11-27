# Copyright (c) 2026 Intel Corporation
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
"""Streaming safetensors shard writer and multimodal VLM text-tower surgery utilities.

Supports exporting large (27B+) text-tower QAT/QAD models back into full multimodal
HuggingFace checkpoints (e.g. ``Qwen3_5ForConditionalGeneration`` with ``model.visual.*``
and ``mtp.*``) while keeping peak host memory bounded by ``<= 2 * max_shard_size``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterator, Optional

import torch


def parse_shard_size_bytes(value: str | int) -> int:
    """Parse human-readable shard sizes like ``'5GB'``, ``'500MB'``, or integer bytes."""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"max_shard_size must be positive, got {value}")
        return value
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)\s*", str(value).lower())
    if match is None:
        raise ValueError(f"Invalid max_shard_size {value!r}")
    num = float(match.group(1))
    unit = match.group(2) or "b"
    scales = {
        "b": 1,
        "kb": 10**3,
        "kib": 2**10,
        "mb": 10**6,
        "mib": 2**20,
        "gb": 10**9,
        "gib": 2**30,
        "tb": 10**12,
        "tib": 2**40,
    }
    if unit not in scales:
        raise ValueError(f"Unsupported unit in max_shard_size {value!r}")
    return max(1, int(num * scales[unit]))


class StreamingSafetensorsWriter:
    """Bounded-memory sharded ``safetensors`` writer.

    Tensors are accumulated up to ``max_shard_size`` bytes and flushed immediately
    to temporary shard files; :meth:`finalize` renames them to HuggingFace canonical
    shard names (``model.safetensors`` or ``model-XXXXX-of-YYYYY.safetensors``) and
    emits ``model.safetensors.index.json`` when ``num_shards > 1``.
    """

    def __init__(self, output_dir: str | Path, max_shard_size: str | int = "5GB") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit_bytes = parse_shard_size_bytes(max_shard_size)
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self._total_bytes = 0
        self._tmp_shards: list[tuple[Path, list[str]]] = []

    def add_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Add ``tensor`` under ``key``, flushing current buffer if shard size would be exceeded."""
        t = tensor.detach().cpu().contiguous()
        nbytes = t.numel() * t.element_size()
        if self._buffer and (self._buffer_bytes + nbytes > self.limit_bytes):
            self._flush_buffer()
        self._buffer[key] = t
        self._buffer_bytes += nbytes
        self._total_bytes += nbytes

    def add_tensors(self, mapping: dict[str, torch.Tensor]) -> None:
        """Add all tensors in ``mapping`` in deterministic key order."""
        for key in sorted(mapping):
            self.add_tensor(key, mapping[key])

    def _flush_buffer(self) -> None:
        from safetensors.torch import save_file

        if not self._buffer:
            return
        idx = len(self._tmp_shards) + 1
        tmp_path = self.output_dir / f".tmp-shard-{idx:05d}.safetensors"
        seen_ptrs: set[int] = set()
        safe_dict: dict[str, torch.Tensor] = {}
        for key, tensor in self._buffer.items():
            ptr = tensor.untyped_storage().data_ptr()
            if ptr in seen_ptrs and tensor.numel() > 0:
                tensor = tensor.clone()
            seen_ptrs.add(tensor.untyped_storage().data_ptr())
            safe_dict[key] = tensor
        save_file(safe_dict, str(tmp_path))
        self._tmp_shards.append((tmp_path, list(safe_dict.keys())))
        self._buffer.clear()
        self._buffer_bytes = 0

    def finalize(self) -> dict[str, str]:
        """Flush remaining buffered tensors and finalize shard names + index file."""
        self._flush_buffer()
        count = len(self._tmp_shards)
        if count == 0:
            return {}

        # Remove stale single/multi-shard files in output_dir if overwriting
        single_file = self.output_dir / "model.safetensors"
        index_file = self.output_dir / "model.safetensors.index.json"
        if count > 1 and single_file.exists():
            single_file.unlink()
        if count == 1 and index_file.exists():
            index_file.unlink()

        weight_map: dict[str, str] = {}
        for idx, (tmp_path, keys) in enumerate(self._tmp_shards, start=1):
            final_name = "model.safetensors" if count == 1 else f"model-{idx:05d}-of-{count:05d}.safetensors"
            final_path = self.output_dir / final_name
            tmp_path.replace(final_path)
            for key in keys:
                weight_map[key] = final_name

        if count > 1:
            index_payload = {
                "metadata": {"total_size": self._total_bytes},
                "weight_map": weight_map,
            }
            index_file.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")

        self._tmp_shards.clear()
        return weight_map


def is_vlm_checkpoint(model_dir: Optional[str | Path], *, vlm_target_prefix: str = "model.language_model.") -> bool:
    """Return ``True`` when ``model_dir`` is a multimodal VLM checkpoint with a nested text tower."""
    if model_dir is None:
        return False
    root = Path(model_dir)
    cfg_path = root / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "vision_config" in cfg or "text_config" in cfg:
            return True
    idx_path = root / "model.safetensors.index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        return any(k.startswith(vlm_target_prefix) for k in idx.get("weight_map", {}))
    return False


def iter_safetensor_shards(model_dir: str | Path) -> list[Path]:
    """Return ordered list of safetensor shard paths inside ``model_dir``."""
    root = Path(model_dir)
    if root.is_file() and root.suffix == ".safetensors":
        return [root]
    idx_path = root / "model.safetensors.index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        seen: list[Path] = []
        seen_names: set[str] = set()
        for shard_name in idx.get("weight_map", {}).values():
            if shard_name not in seen_names:
                seen_names.add(shard_name)
                seen.append(root / shard_name)
        return seen
    single = root / "model.safetensors"
    if single.is_file():
        return [single]
    shards = sorted(p for p in root.glob("model-*.safetensors") if p.is_file())
    if shards:
        return shards
    raise FileNotFoundError(f"No safetensors checkpoint found in {root}")


def iter_safetensors_tensors(model_dir: str | Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream ``(key, cpu_tensor)`` from single or sharded ``safetensors`` in ``model_dir``."""
    from safetensors import safe_open

    for shard_path in iter_safetensor_shards(model_dir):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def remap_text_key_to_vlm(key: str, vlm_target_prefix: str = "model.language_model.") -> str:
    """Remap a standalone CausalLM parameter/module path into a VLM text-tower path."""
    if not vlm_target_prefix:
        return key
    if key == "lm_head" or key.startswith("lm_head.") or key.startswith(vlm_target_prefix):
        return key
    if key == "model":
        return vlm_target_prefix.rstrip(".")
    if key.startswith("model."):
        return vlm_target_prefix + key[len("model.") :]
    return vlm_target_prefix + key


def remap_vlm_key_to_text(key: str, vlm_target_prefix: str = "model.language_model.") -> Optional[str]:
    """Map a VLM checkpoint key to standalone CausalLM key, or ``None`` for non-text keys."""
    if key == "lm_head.weight" or key.startswith("lm_head."):
        return key
    if key.startswith(vlm_target_prefix):
        return "model." + key[len(vlm_target_prefix) :]
    return None


def is_vlm_text_key(key: str, vlm_target_prefix: str = "model.language_model.") -> bool:
    """Return ``True`` if ``key`` belongs to the VLM text tower or ``lm_head``."""
    return key == "lm_head.weight" or key.startswith("lm_head.") or key.startswith(vlm_target_prefix)


def stream_non_text_tensors_from_vlm(
    base_model_path: str | Path,
    writer: StreamingSafetensorsWriter,
    *,
    vlm_target_prefix: str = "model.language_model.",
    dtype: Optional[torch.dtype] = None,
) -> list[str]:
    """Stream non-text tensors (e.g. ``model.visual.*``, ``mtp.*``) from ``base_model_path`` into ``writer``."""
    copied_keys: list[str] = []
    for key, tensor in iter_safetensors_tensors(base_model_path):
        if is_vlm_text_key(key, vlm_target_prefix=vlm_target_prefix):
            continue
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        writer.add_tensor(key, tensor)
        copied_keys.append(key)
    return copied_keys


_AUXILIARY_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def copy_auxiliary_hf_files(src_dir: str | Path, dst_dir: str | Path) -> list[str]:
    """Copy tokenizer, processor, chat template, and generation configs from ``src_dir`` to ``dst_dir``."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    if not src.is_dir():
        return []
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in _AUXILIARY_PATTERNS:
        candidate = src / name
        if candidate.is_file() and candidate.resolve() != (dst / name).resolve():
            shutil.copy2(candidate, dst / name)
            copied.append(name)
    return copied


def normalize_text_config_dict(cfg_dict: dict) -> dict:
    """Normalize standalone text ``config.json`` (e.g. ``Qwen3_5TextModel`` -> ``Qwen3_5ForCausalLM``)."""
    payload = dict(cfg_dict)
    architectures = list(payload.get("architectures") or [])
    if (not architectures or architectures[0].endswith("TextModel")) and payload.get("model_type") == "qwen3_5_text":
        payload["architectures"] = ["Qwen3_5ForCausalLM"]
    return payload


__all__ = [
    "StreamingSafetensorsWriter",
    "copy_auxiliary_hf_files",
    "is_vlm_checkpoint",
    "is_vlm_text_key",
    "iter_safetensor_shards",
    "iter_safetensors_tensors",
    "normalize_text_config_dict",
    "parse_shard_size_bytes",
    "remap_text_key_to_vlm",
    "remap_vlm_key_to_text",
    "stream_non_text_tensors_from_vlm",
]
