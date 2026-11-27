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
"""Unified HuggingFace checkpoint exporter (``compressed-tensors`` & ``bf16`` + VLM surgery)."""

from __future__ import annotations

import copy
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any, Literal, Optional, Union

import torch
import torch.nn as nn

from neural_compressor.torch.quantization.fake_quant import (
    FakeQuantConfig,
    QuantLinear,
    TensorQuantizer,
    apply_amax_map_to_module_config,
    canonicalize_quantizer_key,
    compile_plan_from_info,
    load_amax_map_from_state,
    load_quantizer_state,
    merge_lora_into_quant_linear,
    save_quantizer_state,
)
from neural_compressor.torch.utils import logger

from .compressed_tensors_packer import (
    LayerSchemeSpec,
    build_compressed_tensors_config,
    pack_linear_tensors,
    pack_model_to_compressed_tensors,
)
from .vlm_surgery import (
    StreamingSafetensorsWriter,
    copy_auxiliary_hf_files,
    is_vlm_checkpoint,
    is_vlm_text_key,
    iter_safetensor_shards,
    iter_safetensors_tensors,
    normalize_text_config_dict,
    remap_text_key_to_vlm,
    remap_vlm_key_to_text,
    stream_non_text_tensors_from_vlm,
)

ExportFormat = Literal["compressed_tensors", "bf16"]

_EMBEDDING_SUFFIXES = (
    "embed_tokens",
    "wte",
    "wpe",
    "word_embeddings",
    "position_embeddings",
    "token_embd",
)


def _is_embedding_module_name(mod_name: str) -> bool:
    leaf = mod_name.rsplit(".", 1)[-1] if "." in mod_name else mod_name
    return leaf in _EMBEDDING_SUFFIXES


def _merge_any_lora(model: nn.Module) -> nn.Module:
    model = merge_lora_into_quant_linear(model)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    inner = getattr(model, "peft_model", None)
    if inner is not None and hasattr(inner, "merge_and_unload"):
        model = inner.merge_and_unload()
    return model


def _gather_dense_state_dict(model: nn.Module, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        raw = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False),
        )
    except Exception:
        raw = model.state_dict()
    quantizer_markers = (".weight_quantizer.", ".input_quantizer.", ".output_quantizer.")
    clean: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if any(m in f".{k}" for m in quantizer_markers):
            continue
        t = v.detach()
        if t.is_floating_point():
            t = t.to(dtype=dtype)
        clean[k] = t.cpu().contiguous()
    return clean


def _load_peft_lora_deltas(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, float]], Path]:
    """Load PEFT ``(lora_A, lora_B, scaling)`` keyed by canonical CausalLM module path."""
    from safetensors import safe_open

    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if not adapter_cfg_path.is_file():
        raise FileNotFoundError(f"Expected adapter_config.json in {adapter_dir}")
    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
    r = int(adapter_cfg["r"])
    alpha = float(adapter_cfg.get("lora_alpha", r))
    use_rslora = bool(adapter_cfg.get("use_rslora", False))
    scaling = (alpha / (r**0.5)) if use_rslora else (alpha / r)

    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Expected adapter_model.safetensors in {adapter_dir}")

    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            norm = key
            for pfx in ("base_model.model.",):
                if norm.startswith(pfx):
                    norm = norm[len(pfx) :]
            if norm.endswith(".lora_A.weight"):
                mod = norm[: -len(".lora_A.weight")]
                lora_a[mod] = handle.get_tensor(key)
            elif norm.endswith(".lora_A.default.weight"):
                mod = norm[: -len(".lora_A.default.weight")]
                lora_a[mod] = handle.get_tensor(key)
            elif norm.endswith(".lora_B.weight"):
                mod = norm[: -len(".lora_B.weight")]
                lora_b[mod] = handle.get_tensor(key)
            elif norm.endswith(".lora_B.default.weight"):
                mod = norm[: -len(".lora_B.default.weight")]
                lora_b[mod] = handle.get_tensor(key)

    deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for mod, mat_a in lora_a.items():
        if mod in lora_b:
            deltas[mod] = (mat_a, lora_b[mod], scaling)
    return deltas, adapter_cfg_path


def _discover_quant_config_and_amax(
    ckpt_dir: Path,
    quant_config: Optional[FakeQuantConfig],
    quant_state_path: Optional[Union[str, Path]],
) -> tuple[Optional[FakeQuantConfig], dict[str, float]]:
    state_map: dict[str, float] = {}
    qstate_candidate = Path(quant_state_path) if quant_state_path is not None else (ckpt_dir / "quantizer_state.json")
    if qstate_candidate.is_file():
        state_map.update(load_amax_map_from_state(qstate_candidate))

    qcfg_json = ckpt_dir / "quant_config.json"
    if quant_config is None and qcfg_json.is_file():
        quant_config = FakeQuantConfig.from_json_file(qcfg_json)

    meta_json = ckpt_dir / "quantization_config.json"
    if meta_json.is_file():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        if quant_config is None:
            if isinstance(meta.get("quant"), dict):
                quant_config = FakeQuantConfig.from_dict(meta["quant"])
            elif "config_groups" in meta and meta.get("quant_method") != "compressed-tensors":
                quant_config = FakeQuantConfig.from_dict(meta)
        hex_map = meta.get("amax_hex", {})
        for k, v in meta.get("amax", {}).items():
            val = float.fromhex(hex_map[k]) if k in hex_map else float(v)
            canon_k = canonicalize_quantizer_key(str(k))
            state_map.setdefault(str(k), val)
            state_map.setdefault(canon_k, val)

    if quant_config is not None and quant_config.quantizer_state_path is not None and not state_map:
        p = Path(quant_config.quantizer_state_path)
        if p.is_file():
            state_map.update(load_amax_map_from_state(p))

    return quant_config, state_map


def _scan_text_linear_modules_from_shards(
    source_dir: Path,
    *,
    source_is_vlm: bool,
    vlm_target_prefix: str,
) -> tuple[list[tuple[str, str]], set[str]]:
    """Inspect safetensors headers in <5ms (zero tensor reads) to list text Linear modules."""
    from safetensors import safe_open

    linear_infos: list[tuple[str, str]] = []
    bias_modules: set[str] = set()
    seen: set[str] = set()

    for shard in iter_safetensor_shards(source_dir):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                text_key = remap_vlm_key_to_text(key, vlm_target_prefix) if source_is_vlm else key
                if text_key is None:
                    continue
                if text_key.endswith(".bias"):
                    bias_modules.add(text_key[: -len(".bias")])
                    continue
                if not text_key.endswith(".weight"):
                    continue
                mod_name = text_key[: -len(".weight")]
                if _is_embedding_module_name(mod_name) or mod_name in seen:
                    continue
                shape = handle.get_slice(key).get_shape()
                if len(shape) == 2:
                    seen.add(mod_name)
                    linear_infos.append((mod_name, "Linear"))
    return linear_infos, bias_modules


def _write_final_config_json(
    output_dir: Path,
    *,
    model: Optional[nn.Module],
    source_dir: Optional[Path],
    base_model_path: Optional[Path],
    keep_vlm: bool,
    quant_cfg_dict: Optional[dict[str, Any]],
) -> None:
    cfg_payload: Optional[dict[str, Any]] = None
    if keep_vlm and base_model_path is not None and (base_model_path / "config.json").is_file():
        cfg_payload = json.loads((base_model_path / "config.json").read_text(encoding="utf-8"))
    elif model is not None and getattr(model, "config", None) is not None and hasattr(model.config, "to_dict"):
        cfg_payload = normalize_text_config_dict(model.config.to_dict())
        if not cfg_payload.get("architectures"):
            arch_model = model.get_base_model() if hasattr(model, "get_base_model") else model
            cfg_payload["architectures"] = [type(arch_model).__name__]
    elif source_dir is not None and (source_dir / "config.json").is_file():
        raw_cfg = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
        if not keep_vlm and isinstance(raw_cfg.get("text_config"), dict):
            raw_cfg = raw_cfg["text_config"]
        cfg_payload = normalize_text_config_dict(raw_cfg)
    elif base_model_path is not None and (base_model_path / "config.json").is_file():
        raw_cfg = json.loads((base_model_path / "config.json").read_text(encoding="utf-8"))
        if not keep_vlm and isinstance(raw_cfg.get("text_config"), dict):
            raw_cfg = raw_cfg["text_config"]
        cfg_payload = normalize_text_config_dict(raw_cfg)

    if cfg_payload is None:
        return

    cfg_payload = copy.deepcopy(cfg_payload)
    if quant_cfg_dict is not None:
        cfg_payload["quantization_config"] = quant_cfg_dict
    else:
        cfg_payload.pop("quantization_config", None)

    (output_dir / "config.json").write_text(
        json.dumps(cfg_payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def export_hf_checkpoint(
    model_or_checkpoint: Union[nn.Module, str, Path],
    output_dir: Union[str, Path],
    *,
    export_format: ExportFormat = "compressed_tensors",
    quant_config: Optional[FakeQuantConfig] = None,
    quant_state_path: Optional[Union[str, Path]] = None,
    base_model_path: Optional[Union[str, Path]] = None,
    tokenizer: Any = None,
    keep_vision: bool = True,
    vlm_target_prefix: str = "model.language_model.",
    max_shard_size: str = "5GB",
    dtype: torch.dtype = torch.bfloat16,
    copy_tokenizer_and_processor: bool = True,
) -> Path:
    """Export an in-memory QAT/QAD model or saved checkpoint/adapter directory to HuggingFace format.

    Supports all four workflows with bounded host memory:
    1. In-memory ``QuantLinear`` / ``QuantLoraLinear`` / ``PreTrainedModel``;
    2. Saved full QAT/QAD checkpoint directory (``model.safetensors`` + ``quantizer_state.json``);
    3. Saved PEFT LoRA adapter directory (streaming merge ``W + scaling * B @ A`` with ``base_model_path``);
    4. Multimodal VLM text-tower surgery when ``base_model_path`` is a VLM checkpoint and ``keep_vision=True``.
    """
    if export_format not in ("compressed_tensors", "bf16"):
        raise ValueError(f"export_format must be 'compressed_tensors' or 'bf16', got {export_format!r}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_model_path) if base_model_path is not None else None
    keep_vlm = bool(
        keep_vision and base_path is not None and is_vlm_checkpoint(base_path, vlm_target_prefix=vlm_target_prefix)
    )
    target_prefix = vlm_target_prefix if keep_vlm else ""
    extra_ignore = ["re:.*visual.*", "re:.*mtp.*"] if keep_vlm else None

    writer = StreamingSafetensorsWriter(out, max_shard_size=max_shard_size)
    quant_cfg_dict: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Path A: In-memory nn.Module
    # ------------------------------------------------------------------
    if isinstance(model_or_checkpoint, nn.Module):
        model = _merge_any_lora(model_or_checkpoint)
        if quant_state_path is not None:
            load_quantizer_state(model, quant_state_path, strict=False)

        if export_format == "compressed_tensors":
            if not any(isinstance(m, QuantLinear) for m in model.modules()):
                if quant_config is None:
                    raise ValueError(
                        "Model has no QuantLinear modules; pass a fake-quantized model or provide quant_config."
                    )
                from neural_compressor.torch.quantization.fake_quant import prepare_qat

                prepare_qat(model, quant_config)
                if quant_state_path is not None:
                    load_quantizer_state(model, quant_state_path, strict=False)

            packed_state, quant_cfg_dict = pack_model_to_compressed_tensors(
                model,
                target_prefix=target_prefix,
                extra_ignore=extra_ignore,
                dtype=dtype,
            )
            writer.add_tensors(packed_state)
        else:
            dense_state = _gather_dense_state_dict(model, dtype=dtype)
            for key in sorted(dense_state):
                mapped = remap_text_key_to_vlm(key, vlm_target_prefix) if keep_vlm else key
                writer.add_tensor(mapped, dense_state[key])
            if any(isinstance(m, TensorQuantizer) for m in model.modules()):
                save_quantizer_state(model, out / "quantizer_state.json")

        if keep_vlm and base_path is not None:
            stream_non_text_tensors_from_vlm(
                base_path,
                writer,
                vlm_target_prefix=vlm_target_prefix,
                dtype=dtype if export_format == "bf16" else None,
            )

        writer.finalize()
        _write_final_config_json(
            out,
            model=model,
            source_dir=None,
            base_model_path=base_path,
            keep_vlm=keep_vlm,
            quant_cfg_dict=quant_cfg_dict,
        )
        if copy_tokenizer_and_processor and base_path is not None:
            copy_auxiliary_hf_files(base_path, out)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)
        logger.info(f"[export] wrote {export_format} checkpoint to {out} (keep_vlm={keep_vlm})")
        return out

    # ------------------------------------------------------------------
    # Path B: On-disk Checkpoint Directory or PEFT Adapter Directory
    # ------------------------------------------------------------------
    ckpt_path = Path(model_or_checkpoint)
    if ckpt_path.is_file():
        ckpt_dir = ckpt_path.parent
    else:
        ckpt_dir = ckpt_path

    is_adapter = (ckpt_dir / "adapter_config.json").is_file()
    if is_adapter and base_path is None:
        raise ValueError(f"PEFT adapter directory {ckpt_dir} requires base_model_path to merge base weights.")

    lora_deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    if is_adapter:
        lora_deltas, _ = _load_peft_lora_deltas(ckpt_dir)
        weight_source_dir = base_path
        assert weight_source_dir is not None
        source_is_vlm = is_vlm_checkpoint(weight_source_dir, vlm_target_prefix=vlm_target_prefix)
    else:
        weight_source_dir = ckpt_path if ckpt_path.is_file() else ckpt_dir
        source_is_vlm = is_vlm_checkpoint(ckpt_dir, vlm_target_prefix=vlm_target_prefix)

    quant_config, state_map = _discover_quant_config_and_amax(ckpt_dir, quant_config, quant_state_path)

    plan_dir = base_path if is_adapter else ckpt_dir
    assert plan_dir is not None
    linear_infos, bias_modules = _scan_text_linear_modules_from_shards(
        weight_source_dir,
        source_is_vlm=source_is_vlm,
        vlm_target_prefix=vlm_target_prefix,
    )
    linear_names = {name for name, _ in linear_infos}

    resolved_plan = {}
    if export_format == "compressed_tensors":
        if quant_config is None:
            raise ValueError(
                f"Cannot export {ckpt_dir} to compressed_tensors without FakeQuantConfig "
                "(pass quant_config=... or include quant_config.json in the checkpoint directory)."
            )
        cfg_no_file = copy.deepcopy(quant_config)
        cfg_no_file.quantizer_state_path = None
        report = compile_plan_from_info(linear_infos, cfg_no_file)
        for mod_name, mod_cfg in report.plan.items():
            if state_map:
                apply_amax_map_to_module_config(mod_name, mod_cfg, state_map)
        resolved_plan = report.plan

    # Pre-collect lightweight 1D biases for quantized linear layers so weight+bias pack together cleanly
    pending_biases: dict[str, torch.Tensor] = {}
    if export_format == "compressed_tensors" and bias_modules:
        for raw_key, tensor in iter_safetensors_tensors(weight_source_dir):
            text_key = remap_vlm_key_to_text(raw_key, vlm_target_prefix) if source_is_vlm else raw_key
            if text_key is not None and text_key.endswith(".bias"):
                mod_name = text_key[: -len(".bias")]
                if mod_name in resolved_plan:
                    pending_biases[mod_name] = tensor

    layer_schemes: dict[str, LayerSchemeSpec] = {}
    ignore_targets: list[str] = []
    quantizer_markers = (".weight_quantizer.", ".input_quantizer.", ".output_quantizer.")

    for raw_key, tensor in iter_safetensors_tensors(weight_source_dir):
        if any(m in f".{raw_key}" for m in quantizer_markers):
            continue

        if source_is_vlm and not is_vlm_text_key(raw_key, vlm_target_prefix=vlm_target_prefix):
            if keep_vlm:
                t_out = tensor.to(dtype=dtype) if (export_format == "bf16" and tensor.is_floating_point()) else tensor
                writer.add_tensor(raw_key, t_out)
            continue

        text_key = remap_vlm_key_to_text(raw_key, vlm_target_prefix) if source_is_vlm else raw_key
        assert text_key is not None
        out_key = remap_text_key_to_vlm(text_key, vlm_target_prefix) if keep_vlm else text_key

        # Skip biases already bundled into quantized linear packing
        if export_format == "compressed_tensors" and text_key.endswith(".bias"):
            mod_name = text_key[: -len(".bias")]
            if mod_name in resolved_plan:
                continue

        if text_key.endswith(".weight"):
            mod_name = text_key[: -len(".weight")]
            if mod_name in lora_deltas:
                mat_a, mat_b, scaling = lora_deltas[mod_name]
                mat_b_dev = mat_b.to(device=tensor.device)
                mat_a_dev = mat_a.to(device=tensor.device, dtype=mat_b_dev.dtype)
                delta = (scaling * mat_b_dev) @ mat_a_dev
                target_dtype = tensor.dtype if tensor.is_floating_point() else dtype
                tensor = tensor.to(dtype=target_dtype) + delta.to(dtype=target_dtype)

            if export_format == "compressed_tensors" and mod_name in linear_names:
                mapped_mod = remap_text_key_to_vlm(mod_name, vlm_target_prefix) if keep_vlm else mod_name
                if mod_name in resolved_plan:
                    mod_cfg = resolved_plan[mod_name]
                    wq = TensorQuantizer(mod_cfg.weights)
                    iq = TensorQuantizer(mod_cfg.input_activations)
                    packed_tensors, spec = pack_linear_tensors(
                        mod_name,
                        tensor,
                        pending_biases.pop(mod_name, None),
                        wq,
                        iq,
                        tensor_prefix=mapped_mod,
                        dtype=dtype,
                    )
                    writer.add_tensors(packed_tensors)
                    if spec is not None:
                        layer_schemes[mapped_mod] = spec
                    else:
                        ignore_targets.append(mapped_mod)
                    continue
                else:
                    ignore_targets.append(mapped_mod)

        t_out = tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
        writer.add_tensor(out_key, t_out)

    # When exporting a standalone text checkpoint into a VLM, stream visual/mtp shards from base_path
    if keep_vlm and not source_is_vlm and base_path is not None:
        stream_non_text_tensors_from_vlm(
            base_path,
            writer,
            vlm_target_prefix=vlm_target_prefix,
            dtype=dtype if export_format == "bf16" else None,
        )

    if export_format == "compressed_tensors":
        if extra_ignore:
            ignore_targets.extend(extra_ignore)
        quant_cfg_dict = build_compressed_tensors_config(layer_schemes, ignore_targets=ignore_targets)
    elif state_map:
        payload = {
            "schema": "neural_compressor.quantizer_state/v1",
            "quantizers": {k: {"amax": v, "amax_hex": float(v).hex()} for k, v in sorted(state_map.items())},
        }
        (out / "quantizer_state.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    writer.finalize()
    _write_final_config_json(
        out,
        model=None,
        source_dir=ckpt_dir,
        base_model_path=base_path,
        keep_vlm=keep_vlm,
        quant_cfg_dict=quant_cfg_dict,
    )
    if copy_tokenizer_and_processor:
        if base_path is not None:
            copy_auxiliary_hf_files(base_path, out)
        copy_auxiliary_hf_files(ckpt_dir, out)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(out)

    logger.info(f"[export] streamed {export_format} checkpoint from {ckpt_path} -> {out} (keep_vlm={keep_vlm})")
    return out


def export_compressed_tensors(
    model_or_checkpoint: Union[nn.Module, str, Path],
    output_dir: Union[str, Path],
    *,
    tokenizer: Any = None,
    base_model: Optional[Union[str, Path]] = None,
    quant_config: Optional[FakeQuantConfig] = None,
    quant_state_path: Optional[Union[str, Path]] = None,
    keep_vision: bool = True,
    max_shard_size: str = "5GB",
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Convenience wrapper around :func:`export_hf_checkpoint` for ``compressed_tensors`` export."""
    return export_hf_checkpoint(
        model_or_checkpoint,
        output_dir,
        export_format="compressed_tensors",
        quant_config=quant_config,
        quant_state_path=quant_state_path,
        base_model_path=base_model,
        tokenizer=tokenizer,
        keep_vision=keep_vision,
        max_shard_size=max_shard_size,
        dtype=dtype,
    )


def _export_hf_checkpoint(model: nn.Module, scheme: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exports the torch model to the packed checkpoint with original HF naming.

    Args:
        model: the torch model.
        scheme: the quantization scheme the checkpoint is being exported for, if any.

    Returns:
        post_state_dict: Dict containing quantized weights
        quant_config: config information to export hf_quant_cfg.json
    """
    del scheme
    return pack_model_to_compressed_tensors(model)


def export_hf2compressored_model(
    model: nn.Module,
    export_dir: Path | str = tempfile.gettempdir(),
    scheme: str | None = None,
) -> Path:
    """Backward-compatible alias exporting ``model`` to a ``compressed-tensors`` directory."""
    del scheme
    try:
        return export_hf_checkpoint(model, export_dir, export_format="compressed_tensors")
    except Exception as exc:
        warnings.warn("Cannot export model and config, the state can be saved with torch.save for further inspection.")
        raise exc


__all__ = [
    "ExportFormat",
    "export_compressed_tensors",
    "export_hf2compressored_model",
    "export_hf_checkpoint",
]
