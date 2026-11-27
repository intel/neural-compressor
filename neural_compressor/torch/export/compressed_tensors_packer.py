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
"""Pack fake-quantized linear layers into ``compressed-tensors`` / ``vLLM`` layout.

This module bridges :mod:`neural_compressor.torch.quantization.fake_quant` and
``compressed-tensors`` (>= 0.9) / ``vLLM`` (>= 0.8) without requiring either
inference package to be installed in the training environment:

* ``mxfp4`` -> ``mxfp4-pack-quantized``:
  ``weight_packed`` (``uint8 [out, in // 2]``) + ``weight_scale`` (``uint8 [out, in // 32]`` E8M0).
* ``nvfp4`` -> ``nvfp4-pack-quantized``:
  ``weight_packed`` (``uint8 [out, in // 2]``) + ``weight_scale`` (``float8_e4m3fn [out, in // 16]``)
  + ``weight_global_scale`` (``float32 [1]`` divisor ``1 / global_scale``)
  + optional ``input_global_scale`` (``float32 [1]`` divisor ``(6 * 448) / input_amax``).
* ``mxfp8`` -> ``mxfp8-quantized``:
  ``weight`` (``float8_e4m3fn [out, in]``) + ``weight_scale`` (``uint8 [out, in // 32]`` E8M0).
* ``fp8_e4m3`` -> ``float-quantized``:
  ``weight`` (``float8_e4m3fn [out, in]``) + ``weight_scale`` (``float32 [1]``)
  + optional static ``input_scale`` (``float32 [1]``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from neural_compressor.torch.quantization.fake_quant.formats import E2M1_MAX, E4M3_MAX, get_format
from neural_compressor.torch.quantization.fake_quant.modules import QuantLinear
from neural_compressor.torch.quantization.fake_quant.ops.pack import (
    PackedWeightTensor,
    dequantize_packed_weight,
    pack_weight,
)
from neural_compressor.torch.quantization.fake_quant.quantizer import TensorQuantizer
from neural_compressor.torch.utils import logger


@dataclass(frozen=True)
class LayerSchemeSpec:
    """Canonical ``compressed-tensors`` scheme descriptor for a single Linear module."""

    format: str
    weights: dict[str, Any]
    input_activations: Optional[dict[str, Any]]

    def group_key(self) -> str:
        """Stable JSON key for grouping layers that share identical quantization args."""
        return json.dumps(
            {
                "format": self.format,
                "weights": self.weights,
                "input_activations": self.input_activations,
            },
            sort_keys=True,
        )


def _quant_args_dict(
    *,
    num_bits: int,
    qtype: str = "float",
    symmetric: bool = True,
    strategy: str,
    group_size: Optional[int] = None,
    dynamic: bool | str = False,
    scale_dtype: Optional[str] = None,
    observer: Optional[str] = "memoryless_minmax",
) -> dict[str, Any]:
    return {
        "num_bits": num_bits,
        "type": qtype,
        "symmetric": symmetric,
        "group_size": group_size,
        "strategy": strategy,
        "block_structure": None,
        "dynamic": dynamic,
        "actorder": None,
        "scale_dtype": scale_dtype,
        "zp_dtype": None,
        "observer": observer,
        "observer_kwargs": {},
    }


def _extract_amax_tensor(quantizer: Optional[TensorQuantizer], *, role: str, module_name: str) -> Optional[torch.Tensor]:
    if quantizer is None or not quantizer.is_enabled:
        return None
    amax = quantizer.amax
    if amax is None:
        if not quantizer.is_dynamic:
            raise RuntimeError(
                f"Static {quantizer.format.name} {role}_quantizer on '{module_name}' has no calibrated amax. "
                "Run calibrate_static_amax() or load_quantizer_state() before exporting to compressed-tensors."
            )
        return None
    if hasattr(amax, "full_tensor"):
        amax = amax.full_tensor()
    return amax.detach().to(dtype=torch.float32).cpu().reshape(())


def _fp8_static_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Compute exact 1-element FP32 multiplier ``1 / (448 / amax)`` matching training QDQ."""
    amax_f32 = amax.to(dtype=torch.float32).reshape(())
    safe_amax = torch.where(amax_f32 <= 2.0**-24, torch.ones_like(amax_f32), amax_f32)
    qscale = torch.full_like(safe_amax, E4M3_MAX) / safe_amax
    inv_scale = torch.full_like(qscale, 1.0) / qscale
    return inv_scale.reshape(1).cpu().contiguous()


@torch.no_grad()
def pack_linear_tensors(
    module_name: str,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_quantizer: TensorQuantizer,
    input_quantizer: Optional[TensorQuantizer] = None,
    *,
    tensor_prefix: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, torch.Tensor], Optional[LayerSchemeSpec]]:
    """Pack a single linear module's weight/bias/scales into ``compressed-tensors`` keys.

    Args:
        module_name: Logical module path inside the model (used for diagnostics).
        weight: Dense weight tensor ``[out_features, in_features]`` (already LoRA-merged if needed).
        bias: Optional dense bias tensor ``[out_features]``.
        weight_quantizer: Module's ``weight_quantizer``.
        input_quantizer: Optional module's ``input_quantizer``.
        tensor_prefix: Optional target key prefix (defaults to ``module_name``).
        dtype: Dense fallback dtype for unquantized/bias tensors.

    Returns:
        ``(tensors_dict, scheme_spec)`` where ``scheme_spec`` is ``None`` when the
        layer is unquantized (e.g. ``bf16`` or disabled ``weight_quantizer``).
    """
    prefix = tensor_prefix if tensor_prefix is not None else module_name
    out: dict[str, torch.Tensor] = {}

    if weight_quantizer is None or not weight_quantizer.is_enabled or weight_quantizer.format.name == "bf16":
        out[f"{prefix}.weight"] = weight.detach().to(dtype=dtype).cpu().contiguous()
        if bias is not None:
            out[f"{prefix}.bias"] = bias.detach().to(dtype=dtype).cpu().contiguous()
        return out, None

    fmt = weight_quantizer.format.name
    w_amax = _extract_amax_tensor(weight_quantizer, role="weight", module_name=module_name)
    in_enabled = input_quantizer is not None and input_quantizer.is_enabled and input_quantizer.format.name != "bf16"
    in_amax = _extract_amax_tensor(input_quantizer, role="input", module_name=module_name) if in_enabled else None

    pk = pack_weight(weight.detach(), fmt, amax=w_amax)

    if fmt == "mxfp4":
        assert pk.scale is not None
        out[f"{prefix}.weight_packed"] = pk.data.cpu().contiguous()
        out[f"{prefix}.weight_scale"] = pk.scale.cpu().contiguous()
        w_args = _quant_args_dict(
            num_bits=4,
            strategy="group",
            group_size=32,
            dynamic=False,
            scale_dtype="torch.uint8",
        )
        in_args = (
            _quant_args_dict(
                num_bits=4,
                strategy="group",
                group_size=32,
                dynamic=True,
                scale_dtype="torch.uint8",
                observer=None,
            )
            if in_enabled
            else None
        )
        spec = LayerSchemeSpec(format="mxfp4-pack-quantized", weights=w_args, input_activations=in_args)

    elif fmt == "nvfp4":
        assert pk.scale is not None and pk.global_scale is not None
        out[f"{prefix}.weight_packed"] = pk.data.cpu().contiguous()
        out[f"{prefix}.weight_scale"] = pk.scale.cpu().contiguous()
        # compressed-tensors & vLLM store NVFP4 global scales as DIVISORS: scale_eff = block_scale / global_scale
        w_global_divisor = (1.0 / pk.global_scale.to(torch.float32)).reshape(1).cpu().contiguous()
        out[f"{prefix}.weight_global_scale"] = w_global_divisor

        w_args = _quant_args_dict(
            num_bits=4,
            strategy="tensor_group",
            group_size=16,
            dynamic=False,
            scale_dtype="torch.float8_e4m3fn",
        )
        if in_enabled and in_amax is not None:
            raw_in_global = in_amax.to(torch.float32) / (E2M1_MAX * E4M3_MAX)
            safe_in_global = torch.where(raw_in_global > 0, raw_in_global, torch.ones_like(raw_in_global))
            out[f"{prefix}.input_global_scale"] = (1.0 / safe_in_global).reshape(1).cpu().contiguous()
            in_args = _quant_args_dict(
                num_bits=4,
                strategy="tensor_group",
                group_size=16,
                dynamic="local",
                scale_dtype="torch.float8_e4m3fn",
                observer="static_minmax",
            )
        else:
            if in_enabled and in_amax is None:
                logger.info(
                    f"[export] '{module_name}' uses dynamic NVFP4 input without static global amax; "
                    "exporting weight-only NVFP4A16 scheme for safe vLLM execution."
                )
            in_args = None
        spec = LayerSchemeSpec(format="nvfp4-pack-quantized", weights=w_args, input_activations=in_args)

    elif fmt == "mxfp8":
        assert pk.scale is not None
        out[f"{prefix}.weight"] = pk.data.cpu().contiguous()
        out[f"{prefix}.weight_scale"] = pk.scale.cpu().contiguous()
        w_args = _quant_args_dict(
            num_bits=8,
            strategy="group",
            group_size=32,
            dynamic=False,
            scale_dtype="torch.uint8",
        )
        in_args = (
            _quant_args_dict(
                num_bits=8,
                strategy="group",
                group_size=32,
                dynamic=True,
                scale_dtype="torch.uint8",
                observer=None,
            )
            if in_enabled
            else None
        )
        spec = LayerSchemeSpec(format="mxfp8-quantized", weights=w_args, input_activations=in_args)

    elif fmt == "fp8_e4m3":
        assert pk.scale is not None
        out[f"{prefix}.weight"] = pk.data.cpu().contiguous()
        out[f"{prefix}.weight_scale"] = pk.scale.to(torch.float32).reshape(1).cpu().contiguous()
        w_args = _quant_args_dict(
            num_bits=8,
            strategy="tensor",
            dynamic=False,
        )
        if in_enabled:
            assert input_quantizer is not None
            if not input_quantizer.is_dynamic:
                assert in_amax is not None
                out[f"{prefix}.input_scale"] = _fp8_static_scale_from_amax(in_amax)
                in_args = _quant_args_dict(
                    num_bits=8,
                    strategy="tensor",
                    dynamic=False,
                    observer="static_minmax",
                )
            else:
                in_args = _quant_args_dict(
                    num_bits=8,
                    strategy="token",
                    dynamic=True,
                    observer=None,
                )
        else:
            in_args = None
        spec = LayerSchemeSpec(format="float-quantized", weights=w_args, input_activations=in_args)

    else:
        raise ValueError(f"Unsupported weight quantizer format {fmt!r} on '{module_name}'")

    if bias is not None:
        out[f"{prefix}.bias"] = bias.detach().to(dtype=dtype).cpu().contiguous()

    return out, spec


def build_compressed_tensors_config(
    layer_schemes: dict[str, LayerSchemeSpec],
    *,
    ignore_targets: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Assemble ``config.json['quantization_config']`` in ``compressed-tensors`` schema.

    ``targets`` in each ``config_group`` are always expanded to sorted concrete module
    paths (never broad regexes such as ``re:.*mlp.*``) so vision-tower MLPs or GDN
    auxiliary projections in hybrid/multimodal architectures are never accidentally matched.
    """
    if not layer_schemes:
        raise ValueError("Cannot build compressed-tensors quantization_config: 0 quantized layers found.")

    grouped_targets: dict[str, list[str]] = {}
    key_to_spec: dict[str, LayerSchemeSpec] = {}
    for target_name in sorted(layer_schemes):
        spec = layer_schemes[target_name]
        gkey = spec.group_key()
        grouped_targets.setdefault(gkey, []).append(target_name)
        key_to_spec[gkey] = spec

    # Deterministic ordering: sort groups by (format, first target name)
    ordered_keys = sorted(grouped_targets, key=lambda k: (key_to_spec[k].format, grouped_targets[k][0]))
    config_groups: dict[str, dict[str, Any]] = {}
    formats_seen: set[str] = set()

    for idx, gkey in enumerate(ordered_keys):
        spec = key_to_spec[gkey]
        formats_seen.add(spec.format)
        config_groups[f"group_{idx}"] = {
            "targets": sorted(grouped_targets[gkey]),
            "weights": spec.weights,
            "input_activations": spec.input_activations,
            "output_activations": None,
            "format": spec.format,
        }

    top_format = next(iter(formats_seen)) if len(formats_seen) == 1 else "mixed-precision"
    dedup_ignore = list(dict.fromkeys(ignore_targets or []))

    quant_cfg: dict[str, Any] = {
        "config_groups": config_groups,
        "format": top_format,
        "global_compression_ratio": None,
        "ignore": dedup_ignore,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    # Optional zero-dependency validation hook if compressed_tensors is installed
    try:
        from compressed_tensors.quantization import QuantizationConfig as CTQuantizationConfig

        CTQuantizationConfig.model_validate(quant_cfg)
    except ImportError:
        pass

    return quant_cfg


@torch.no_grad()
def dequantize_compressed_tensors_layer(
    prefix: str,
    tensors: dict[str, torch.Tensor],
    scheme_format: str,
    *,
    original_shape: Optional[tuple[int, int]] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an exported ``compressed-tensors`` layer back to dense floating-point."""
    if scheme_format == "mxfp4-pack-quantized":
        w_packed = tensors[f"{prefix}.weight_packed"]
        w_scale = tensors[f"{prefix}.weight_scale"]
        shape = original_shape or (w_packed.shape[0], w_packed.shape[1] * 2)
        pk = PackedWeightTensor(
            data=w_packed,
            scale=w_scale,
            global_scale=None,
            original_shape=shape,
            format=get_format("mxfp4"),
        )
        return dequantize_packed_weight(pk, dtype=dtype)

    if scheme_format == "nvfp4-pack-quantized":
        w_packed = tensors[f"{prefix}.weight_packed"]
        w_scale = tensors[f"{prefix}.weight_scale"]
        w_global_divisor = tensors[f"{prefix}.weight_global_scale"]
        shape = original_shape or (w_packed.shape[0], w_packed.shape[1] * 2)
        pk = PackedWeightTensor(
            data=w_packed,
            scale=w_scale,
            global_scale=(1.0 / w_global_divisor.to(torch.float32)).reshape(1),
            original_shape=shape,
            format=get_format("nvfp4"),
        )
        return dequantize_packed_weight(pk, dtype=dtype)

    if scheme_format == "mxfp8-quantized":
        w_q = tensors[f"{prefix}.weight"]
        w_scale = tensors[f"{prefix}.weight_scale"]
        shape = original_shape or (w_q.shape[0], w_q.shape[1])
        pk = PackedWeightTensor(
            data=w_q,
            scale=w_scale,
            global_scale=None,
            original_shape=shape,
            format=get_format("mxfp8"),
        )
        return dequantize_packed_weight(pk, dtype=dtype)

    if scheme_format == "float-quantized":
        w_q = tensors[f"{prefix}.weight"]
        w_scale = tensors[f"{prefix}.weight_scale"]
        shape = original_shape or (w_q.shape[0], w_q.shape[1])
        pk = PackedWeightTensor(
            data=w_q,
            scale=w_scale,
            global_scale=None,
            original_shape=shape,
            format=get_format("fp8_e4m3"),
        )
        return dequantize_packed_weight(pk, dtype=dtype)

    raise ValueError(f"Unsupported compressed-tensors scheme format {scheme_format!r}")


@torch.no_grad()
def pack_model_to_compressed_tensors(
    model: nn.Module,
    *,
    target_prefix: str = "",
    extra_ignore: Optional[list[str]] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Pack an in-memory ``QuantLinear`` model into ``(compressed_state_dict, quantization_config)``.

    Any ``QuantLoraLinear`` or PEFT wrapper on ``model`` should be merged beforehand via
    :func:`merge_lora_into_quant_linear` (handled automatically by ``export_hf_checkpoint``).
    """
    from neural_compressor.torch.quantization.fake_quant import merge_lora_into_quant_linear

    model = merge_lora_into_quant_linear(model)

    # Gather full model state dict safely if FSDP2 sharded
    raw_state: dict[str, torch.Tensor]
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        raw_state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False),
        )
    except Exception:
        raw_state = model.state_dict()

    def _map_module_name(name: str) -> str:
        if not target_prefix:
            return name
        if name == "lm_head" or name.startswith("lm_head."):
            return name
        if name.startswith("model."):
            return target_prefix + name[len("model.") :]
        return target_prefix + name

    def _map_tensor_key(key: str) -> str:
        if not target_prefix:
            return key
        if key == "lm_head.weight" or key.startswith("lm_head."):
            return key
        if key.startswith("model."):
            return target_prefix + key[len("model.") :]
        return target_prefix + key

    packed_state: dict[str, torch.Tensor] = {}
    layer_schemes: dict[str, LayerSchemeSpec] = {}
    ignore_targets: list[str] = []
    handled_module_prefixes: set[str] = set()

    for mod_name, mod in model.named_modules():
        if isinstance(mod, QuantLinear):
            mapped_mod = _map_module_name(mod_name)
            w_key = f"{mod_name}.weight" if mod_name else "weight"
            b_key = f"{mod_name}.bias" if mod_name else "bias"
            weight = raw_state.get(w_key, mod.weight)
            bias = raw_state.get(b_key, mod.bias) if mod.bias is not None else None
            tensors, spec = pack_linear_tensors(
                mod_name,
                weight,
                bias,
                mod.weight_quantizer,
                mod.input_quantizer,
                tensor_prefix=mapped_mod,
                dtype=dtype,
            )
            packed_state.update(tensors)
            handled_module_prefixes.add(mod_name)
            if spec is not None:
                layer_schemes[mapped_mod] = spec
            else:
                ignore_targets.append(mapped_mod)
        elif isinstance(mod, nn.Linear):
            mapped_mod = _map_module_name(mod_name)
            ignore_targets.append(mapped_mod)

    # Copy all remaining non-quantizer parameters/buffers (embeddings, norms, unquantized linears, etc.)
    quantizer_markers = (".weight_quantizer.", ".input_quantizer.", ".output_quantizer.")
    for key, val in raw_state.items():
        if any(m in f".{key}" for m in quantizer_markers):
            continue
        mod_prefix = key.rsplit(".", 1)[0] if "." in key else ""
        if mod_prefix in handled_module_prefixes:
            continue
        mapped_key = _map_tensor_key(key)
        tensor = val.detach()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        packed_state[mapped_key] = tensor.cpu().contiguous()

    if extra_ignore:
        ignore_targets.extend(extra_ignore)

    quant_cfg = build_compressed_tensors_config(layer_schemes, ignore_targets=ignore_targets)
    return packed_state, quant_cfg


__all__ = [
    "LayerSchemeSpec",
    "build_compressed_tensors_config",
    "dequantize_compressed_tensors_layer",
    "pack_linear_tensors",
    "pack_model_to_compressed_tensors",
]
