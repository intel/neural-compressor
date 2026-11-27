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
"""Load Qwen3.5 / Qwen3.8 VLM or CausalLM checkpoints as standard ``PreTrainedModel``.

When given a multimodal ``Qwen3_5ForConditionalGeneration`` directory (whose safetensors keys
are ``model.language_model.*``, ``model.visual.*``, and ``lm_head.weight``), this helper streams
only the text tower into a standard ``transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM``
instance, skipping the vision tower during text-only QAD while remaining 100% compatible with
``peft.get_peft_model``, FSDP2, ``QATTrainer``, ``KDTrainer``, ``QADTrainer``, and
``export_hf_checkpoint(..., base_model_path=vlm_dir, keep_vision=True)``.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_cpu_safe_qwen35_fallbacks(model: nn.Module) -> nn.Module:
    """Allow ``Qwen3_5GatedDeltaNet`` to fall back cleanly to PyTorch ops when tensors live on CPU.

    When ``flash-linear-attention`` / ``causal-conv1d`` are installed, ``Qwen3_5GatedDeltaNet`` binds
    CUDA/Triton kernels unconditionally at ``__init__`` time. Wrapping those hooks with an ``is_cuda``
    fast-path check preserves 100% native kernel speed on GPU while supporting CPU smoke tests and
    ``--kd_teacher_device cpu``.
    """
    present = {type(m).__name__ for m in model.modules()}
    if hasattr(model, "_no_split_modules") and model._no_split_modules:
        model._no_split_modules = [n for n in model._no_split_modules if n in present]
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        Qwen3_5RMSNormGated,
        torch_causal_conv1d_update,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    for mod in model.modules():
        if not isinstance(mod, Qwen3_5GatedDeltaNet):
            continue
        fast_chunk = mod.chunk_gated_delta_rule
        if fast_chunk is not torch_chunk_gated_delta_rule:
            def _chunk_dispatch(query, key, value, g, beta, _fast=fast_chunk, **kwargs):
                if query.is_cuda:
                    return _fast(query, key, value, g=g, beta=beta, **kwargs)
                return torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, **kwargs)

            mod.chunk_gated_delta_rule = _chunk_dispatch

        fast_recurrent = mod.recurrent_gated_delta_rule
        if fast_recurrent is not torch_recurrent_gated_delta_rule:
            def _recurrent_dispatch(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                _fast=fast_recurrent,
            ):
                if query.is_cuda:
                    return _fast(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    )
                return torch_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )

            mod.recurrent_gated_delta_rule = _recurrent_dispatch

        if mod.causal_conv1d_fn is not None:
            fast_conv = mod.causal_conv1d_fn

            def _conv_dispatch(*, x, weight, bias=None, activation="silu", seq_idx=None, _fast=fast_conv, _mod=mod):
                if x.is_cuda:
                    return _fast(x=x, weight=weight, bias=bias, activation=activation, seq_idx=seq_idx)
                return F.silu(_mod.conv1d(x)[:, :, : x.shape[-1]])

            mod.causal_conv1d_fn = _conv_dispatch

        fast_conv_update = mod.causal_conv1d_update
        if fast_conv_update is not torch_causal_conv1d_update:
            def _conv_update_dispatch(x, conv_state, weight, bias=None, activation=None, _fast=fast_conv_update):
                if x.is_cuda:
                    return _fast(x, conv_state, weight, bias, activation)
                return torch_causal_conv1d_update(x, conv_state, weight, bias, activation)

            mod.causal_conv1d_update = _conv_update_dispatch

        if not isinstance(mod.norm, Qwen3_5RMSNormGated):
            norm_mod = mod.norm
            fast_norm_fwd = norm_mod.forward
            eps = float(getattr(norm_mod, "eps", getattr(norm_mod, "variance_epsilon", 1e-6)))

            def _norm_dispatch(hidden_states, gate=None, _fast=fast_norm_fwd, _norm=norm_mod, _eps=eps):
                if hidden_states.is_cuda:
                    return _fast(hidden_states, gate)
                in_dtype = hidden_states.dtype
                hs = hidden_states.to(torch.float32)
                hs = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + _eps)
                hs = _norm.weight * hs.to(in_dtype)
                if gate is not None:
                    hs = hs * F.silu(gate.to(torch.float32))
                return hs.to(in_dtype)

            norm_mod.forward = _norm_dispatch
    return model


def build_tiny_qwen35(
    *,
    vocab_size: int = 128,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 16,
    layer_types: Optional[List[str]] = None,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Build a tiny standard ``Qwen3_5ForCausalLM`` model for smoke runs and unit tests."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    if layer_types is None:
        layer_types = ["linear_attention", "full_attention"][:num_hidden_layers]
        if len(layer_types) < num_hidden_layers:
            layer_types = layer_types + ["linear_attention"] * (num_hidden_layers - len(layer_types))

    config = Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=num_key_value_heads,
        linear_num_value_heads=num_attention_heads,
        layer_types=layer_types,
        tie_word_embeddings=False,
        dtype=dtype,
    )
    model = Qwen3_5ForCausalLM(config).to(device="cpu", dtype=dtype)
    return _install_cpu_safe_qwen35_fallbacks(model)


def _resolve_checkpoint_dir(path: str) -> Path:
    root = Path(path)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(  # nosec B615 - caller-supplied model
                path,
                allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab*", "merges*"],
            )
        )
    return root


def _finalize_meta_loaded_buffers(model: nn.Module, loaded_keys: set[str]) -> None:
    """Mark loaded tensors initialized and recompute non-persistent buffers (e.g. RoPE ``inv_freq``).

    Initializing on ``meta`` + ``to_empty(device="cpu")`` leaves non-persistent buffers uninitialized;
    calling ``initialize_weights()`` after setting ``_is_hf_initialized = True`` on checkpoint-backed
    tensors recomputes RoPE frequencies without disturbing loaded weights.
    """
    for param in model.parameters():
        param._is_hf_initialized = True
    for name, buf in model.named_buffers():
        if name in loaded_keys:
            buf._is_hf_initialized = True
    model.initialize_weights()


def _load_qwen35_vlm_text_as_causal_lm(path: str, vlm_config, dtype: torch.dtype) -> nn.Module:
    """Stream ``model.language_model.*`` + ``lm_head.*`` from a Qwen3.5 VLM checkpoint into ``Qwen3_5ForCausalLM``."""
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    text_config = vlm_config.text_config
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(text_config)
    model = model.to_empty(device="cpu")

    root = _resolve_checkpoint_dir(path)
    index_file = root / "model.safetensors.index.json"
    if index_file.is_file():
        weight_map = json.loads(index_file.read_text())["weight_map"]
        relevant_shards = sorted(
            {
                root / shard_name
                for key, shard_name in weight_map.items()
                if key.startswith("model.language_model.") or key.startswith("lm_head.")
            }
        )
    else:
        relevant_shards = sorted(root.glob("*.safetensors"))
    if not relevant_shards:
        raise FileNotFoundError(f"No safetensors shards found under {path}")

    loaded_keys: set[str] = set()
    for shard in relevant_shards:
        shard_state: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("model.language_model."):
                    mapped = "model." + key[len("model.language_model.") :]
                    shard_state[mapped] = handle.get_tensor(key).to(dtype)
                    loaded_keys.add(mapped)
                elif key.startswith("lm_head."):
                    shard_state[key] = handle.get_tensor(key).to(dtype)
                    loaded_keys.add(key)
        if shard_state:
            model.load_state_dict(shard_state, strict=False, assign=True)
        del shard_state
        gc.collect()

    if not loaded_keys:
        raise ValueError(f"{path}: no `model.language_model.*` tensors found in checkpoint")

    _finalize_meta_loaded_buffers(model, loaded_keys)
    return _install_cpu_safe_qwen35_fallbacks(model)


def load_qwen_causal_lm(
    path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    tiny: bool = False,
    trust_remote_code: bool = False,
    use_unsloth: bool = False,
    max_seq_length: int = 8192,
) -> nn.Module:
    """Load a standard ``PreTrainedModel`` for text-only QAD training or calibration."""
    if tiny:
        return build_tiny_qwen35(dtype=dtype)

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise RuntimeError("`use_unsloth=True` requires the optional `unsloth` package.") from exc

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model, _ = FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
            load_in_16bit=True,
            text_only=True,
            trust_remote_code=trust_remote_code,
            device_map=f"cuda:{local_rank}",
        )
        for param in model.parameters():
            param.requires_grad_(True)
        return model

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)  # nosec B615 - caller-supplied model
    if getattr(config, "model_type", "") == "qwen3_5" and hasattr(config, "text_config"):
        return _load_qwen35_vlm_text_as_causal_lm(path, config, dtype)

    return AutoModelForCausalLM.from_pretrained(  # nosec B615 - caller-supplied model
        path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        device_map="cpu",
    )


__all__ = ["build_tiny_qwen35", "load_qwen_causal_lm"]
