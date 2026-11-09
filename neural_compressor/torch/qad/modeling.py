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
"""Model loading and the model-adapter registry.

An adapter answers *structural* questions only (where is the text tower, which
modules are decoder layers, which one is ``lm_head``, what should FSDP2 wrap).
Quantization policy lives in recipes; nothing model-specific may leak into the
mechanism layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from neural_compressor.torch.utils import logger


class CausalLM(nn.Module):
    """Text tower + ``lm_head``; the shape the QAD trainer consumes.

    ``HF`` causal LMs already look like this (``model.model`` + ``model.lm_head``),
    but a VLM checkpoint only exposes ``model.language_model``. This container
    gives both layouts one interface so the trainer never branches on
    architecture.
    """

    def __init__(self, text_model: nn.Module, lm_head: nn.Module, config: Any) -> None:
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head
        self.config = config

    @property
    def layers(self) -> nn.ModuleList:
        """Decoder layer list of the text tower."""
        layers = getattr(self.model, "layers", None)
        if layers is None:
            raise AttributeError(f"{type(self.model).__name__} has no .layers")
        return layers

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the text tower and return its last hidden state."""
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # FSDP2 registers pre-forward hooks on forward(); the trainer must call
        # this path so the root module is unsharded before embedding/lm_head.
        return self.forward_hidden(input_ids, attention_mask, position_ids)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply ``lm_head`` (cast to its dtype: hybrid models emit fp32 states)."""
        return self.lm_head(hidden.to(self.lm_head.weight.dtype))

    def gradient_checkpointing_enable(self, **kwargs):
        """Forward to the text tower when it supports checkpointing."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)
        return None

    def gradient_checkpointing_disable(self):
        """Forward to the text tower when it supports checkpointing."""
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()
        return None

    def num_parameters(
        self,
        only_trainable: bool = False,
        exclude_embeddings: bool = False,
        trainable_only: Optional[bool] = None,
    ) -> int:
        """Count parameters with optional trainable/embedding filters."""
        if trainable_only is not None:
            only_trainable = trainable_only
        total = 0
        for name, param in self.named_parameters():
            if only_trainable and not param.requires_grad:
                continue
            if exclude_embeddings and ("embed_tokens" in name or name.endswith("lm_head.weight")):
                continue
            total += param.numel()
        return total


@runtime_checkable
class ModelAdapter(Protocol):
    """Structural access to a causal LM (see module docstring)."""

    family: str

    def text_model(self, model: nn.Module) -> nn.Module:
        """Return the transformer stack (without ``lm_head``)."""

    def layers(self, model: nn.Module) -> nn.ModuleList:
        """Return the decoder layer list."""

    def lm_head(self, model: nn.Module) -> nn.Module:
        """Return the output projection."""

    def wrap_classes(self, model: nn.Module) -> List[str]:
        """Return decoder-layer class names for FSDP2 auto-wrap."""


class GenericAdapter:
    """Duck-typed adapter covering standard HF decoder stacks and :class:`CausalLM`.

    Mirrors ModelOpt's ``get_homogeneous_hf_decoder_layers`` fallbacks
    (``model.model`` / ``model.language_model`` / ``.layers``) so unseen
    decoder-only checkpoints need no registration.
    """

    family = "generic"

    def text_model(self, model: nn.Module) -> nn.Module:
        """Locate the transformer stack."""
        if isinstance(model, CausalLM):
            return model.model
        if hasattr(model, "language_model"):
            return model.language_model
        return getattr(model, "model", model)

    def layers(self, model: nn.Module) -> nn.ModuleList:
        """Locate the decoder layer list."""
        text = self.text_model(model)
        layers = getattr(text, "layers", None)
        if layers is None:
            raise AttributeError(
                f"cannot find decoder layers on {type(model).__name__}; " "register a ModelAdapter for this family"
            )
        return layers

    def lm_head(self, model: nn.Module) -> nn.Module:
        """Locate the output projection."""
        head = getattr(model, "lm_head", None)
        if head is None:
            raise AttributeError(f"{type(model).__name__} has no lm_head")
        return head

    def wrap_classes(self, model: nn.Module) -> List[str]:
        """Unique decoder-layer class names (what FSDP2 wraps)."""
        return sorted({type(layer).__name__ for layer in self.layers(model)})


class Qwen35Adapter(GenericAdapter):
    """Qwen3.5 (text or VLM): text tower sits at ``model.language_model``."""

    family = "qwen3_5"


_ADAPTERS: Dict[str, ModelAdapter] = {
    "generic": GenericAdapter(),
    "qwen3_5": Qwen35Adapter(),
    "qwen3_5_text": Qwen35Adapter(),
}


def register_adapter(family: str, adapter: ModelAdapter) -> ModelAdapter:
    """Register an adapter for a ``model_type``."""
    _ADAPTERS[family] = adapter
    return adapter


def get_adapter(model_or_family: Any) -> ModelAdapter:
    """Resolve an adapter from a model, a config or a family string."""
    family = model_or_family if isinstance(model_or_family, str) else None
    if family is None:
        config = getattr(model_or_family, "config", None)
        family = getattr(config, "model_type", None)
    adapter = _ADAPTERS.get(family or "generic")
    if adapter is None:
        logger.warning(f"no adapter registered for model_type={family!r}; using generic")
        adapter = _ADAPTERS["generic"]
    return adapter


def load_model(
    path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    tiny: bool = False,
    trust_remote_code: bool = False,
) -> nn.Module:
    """Load a student/teacher for QAD.

    Text-only checkpoints use ``AutoModelForCausalLM``. Qwen3.5 VLMs load only
    the language tower (``model.language_model.*`` + ``lm_head``) to avoid
    materializing the vision encoder during training.
    """
    if tiny:
        return build_tiny_qwen35(dtype=dtype)

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", "")
    if model_type == "qwen3_5" and hasattr(config, "text_config"):
        return _load_qwen35_text(path, config, dtype)
    full = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        device_map="cpu",
    )
    text = getattr(full, "model", None)
    head = getattr(full, "lm_head", None)
    if text is None or head is None:
        raise ValueError(f"{path}: cannot locate .model/.lm_head on {type(full).__name__}")
    return CausalLM(text, head, full.config)


def hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    adapter: Optional[ModelAdapter] = None,
) -> torch.Tensor:
    """Run the text tower through the root module's ``forward``.

    Going through ``forward`` (instead of calling ``text_model(...)`` directly)
    is required under FSDP2: its pre-forward hook unshards the root, and the
    embedding/lm_head live there.
    """
    if hasattr(model, "forward_hidden"):
        return model.forward_hidden(input_ids, attention_mask)
    text = (adapter or get_adapter(model)).text_model(model)
    out = text(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def logits_from_hidden(
    model: nn.Module,
    hidden: torch.Tensor,
    adapter: Optional[ModelAdapter] = None,
) -> torch.Tensor:
    """Apply the output projection to a hidden-state chunk."""
    if hasattr(model, "logits"):
        return model.logits(hidden)
    head = (adapter or get_adapter(model)).lm_head(model)
    return head(hidden.to(head.weight.dtype))


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
) -> CausalLM:
    """Build a tiny Qwen3.5 text model for CPU tests (one GDN + one attention)."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    if layer_types is None:
        # The reference Qwen3.8 tower is GDN-heavy (full attention every 4th
        # layer); default to GDN so layer indices line up with the real recipe.
        layer_types = ["linear_attention"] * num_hidden_layers
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
    text = Qwen3_5TextModel(config).to(dtype)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)
    return CausalLM(text, lm_head, config)


def _load_qwen35_text(path: str, config, dtype: torch.dtype) -> CausalLM:
    """Load only the language tower of a Qwen3.5 VLM checkpoint.

    Checkpoint keys are ``model.language_model.*`` and ``lm_head.weight``.
    Non-persistent buffers (RoPE ``inv_freq``) must be re-initialized: a meta
    init plus ``to_empty`` leaves them as uninitialized memory, which produced
    the FSDP2 NaN investigated in the QAD experiments.
    """
    import gc
    import json
    from pathlib import Path

    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    text_config = config.text_config
    with torch.device("meta"):
        text = Qwen3_5TextModel(text_config)
        lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False, dtype=dtype)
    text = text.to_empty(device="cpu")
    lm_head = lm_head.to_empty(device="cpu")

    root = Path(path)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        files = sorted({root / name for name in json.loads(index.read_text())["weight_map"].values()})
    else:
        files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")

    text_state, head_state = {}, {}
    for shard in files:
        with safe_open(str(shard), "pt") as handle:
            for key in handle.keys():
                if key.startswith("model.language_model."):
                    text_state[key[len("model.language_model.") :]] = handle.get_tensor(key).to(dtype)
                elif key.startswith("lm_head."):
                    head_state[key[len("lm_head.") :]] = handle.get_tensor(key).to(dtype)
    if not text_state:
        raise ValueError(f"{path}: no model.language_model.* tensors found")
    text.load_state_dict(text_state, strict=False, assign=True)
    lm_head.load_state_dict(head_state, strict=False, assign=True)
    _init_non_persistent_buffers(text, set(text_state))
    del text_state, head_state
    gc.collect()
    return CausalLM(text, lm_head, text_config)


def _init_non_persistent_buffers(model: nn.Module, loaded_keys: set) -> None:
    """Recompute buffers that meta init + ``to_empty`` left uninitialized.

    Mirrors ``PreTrainedModel._finalize_model_loading``: mark tensors that came
    from the checkpoint, then let ``initialize_weights()`` rewrite the rest
    (RoPE frequencies are non-persistent and never in the checkpoint).
    """
    for param in model.parameters():
        param._is_hf_initialized = True
    for name, buffer in model.named_buffers():
        if name in loaded_keys:
            buffer._is_hf_initialized = True
    model.initialize_weights()


@dataclass
class LoadReport:
    """What :func:`load_model` found, for logs and tests."""

    family: str
    n_layers: int
    hidden_size: int
    vocab_size: int
    wrap_classes: List[str]


__all__ = [
    "CausalLM",
    "GenericAdapter",
    "LoadReport",
    "ModelAdapter",
    "Qwen35Adapter",
    "build_tiny_qwen35",
    "get_adapter",
    "hidden_states",
    "logits_from_hidden",
    "load_model",
    "register_adapter",
]
