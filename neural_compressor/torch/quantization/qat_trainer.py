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
"""Standalone HuggingFace ``Trainer`` subclass for Quantization-Aware Training (QAT)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import Trainer, TrainingArguments

from neural_compressor.torch.utils import logger

from .fake_quant import (
    FakeQuantConfig,
    TensorQuantizer,
    calibrate_static_amax,
    load_quantizer_state,
    merge_lora_into_quant_linear,
    save_quantizer_state,
)
from .quantize import prepare_qat


def unwrap_model(model: Any) -> nn.Module:
    """Unwrap DDP / FSDP ``.module`` wrappers to access the underlying ``nn.Module``."""
    while hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
        model = model.module
    return model


def is_peft_model(model: nn.Module) -> bool:
    """Return True when ``model`` is a PEFT ``PeftModel`` wrapper."""
    inner = unwrap_model(model)
    return hasattr(inner, "peft_config") and hasattr(inner, "save_pretrained")


def ensure_peft_gradient_checkpointing_delegation(model: nn.Module) -> None:
    """Ensure ``PeftModel`` forwards gradient checkpointing toggles to the inner backbone."""
    if not (hasattr(model, "peft_config") and hasattr(model, "base_model")):
        return
    inner = getattr(model.base_model, "model", None)
    if inner is None:
        return

    if not hasattr(model, "_nc_gc_patched"):
        orig_enable = getattr(model, "gradient_checkpointing_enable", None)
        orig_disable = getattr(model, "gradient_checkpointing_disable", None)

        def _gc_enable(**kwargs):
            if hasattr(inner, "gradient_checkpointing_enable"):
                return inner.gradient_checkpointing_enable(**kwargs)
            if callable(orig_enable):
                return orig_enable(**kwargs)
            return None

        def _gc_disable():
            if hasattr(inner, "gradient_checkpointing_disable"):
                return inner.gradient_checkpointing_disable()
            if callable(orig_disable):
                return orig_disable()
            return None

        model.gradient_checkpointing_enable = _gc_enable  # type: ignore[method-assign]
        model.gradient_checkpointing_disable = _gc_disable  # type: ignore[method-assign]
        object.__setattr__(model, "_nc_gc_patched", True)


def _without_quantizer_buffers(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    slots = (".weight_quantizer.", ".input_quantizer.", ".output_quantizer.")
    return {k: v for k, v in state.items() if not any(slot in k for slot in slots)}


def save_fsdp2_peft_adapter(model: nn.Module, out_dir: Union[str, Path], *, should_save: bool) -> bool:
    """Gather sharded PEFT LoRA adapter parameters under FSDP2 and save a PEFT checkpoint."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    state = get_model_state_dict(model)
    adapter_state = {key: value for key, value in state.items() if ".lora_" in key}
    if not adapter_state:
        return False
    gathered = {
        key: (value.full_tensor().cpu() if hasattr(value, "full_tensor") else value.detach().cpu())
        for key, value in adapter_state.items()
    }
    if should_save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        unwrap_model(model).save_pretrained(out, state_dict=gathered)
    return True


def save_fsdp2_full_model(
    model: nn.Module,
    out_dir: Union[str, Path],
    *,
    should_save: bool,
    dtype: Optional[torch.dtype] = None,
) -> bool:
    """Gather a full fine-tuned model under FSDP2 to CPU rank 0 and write ``model.safetensors``."""
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    target = merge_lora_into_quant_linear(unwrap_model(model)) if is_peft_model(model) else unwrap_model(model)
    state = get_model_state_dict(
        target,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False),
    )
    if should_save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        clean_state = _without_quantizer_buffers(state)
        tensors = {}
        for key, value in clean_state.items():
            t = value.detach()
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype=dtype)
            tensors[key] = t.cpu().contiguous()
        save_file(tensors, str(out / "model.safetensors"))
        cfg = getattr(target, "config", None)
        if cfg is not None and hasattr(cfg, "to_dict"):
            (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8")
    return True


def _prune_absent_no_split_modules(model: nn.Module) -> None:
    """Filter ``_no_split_modules`` on ``model`` (and inner PEFT modules) to classes actually present.

    Multimodal-derived ``ForCausalLM`` classes in ``transformers`` (e.g. ``Qwen3_5ForCausalLM``) often
    inherit vision block names in ``_no_split_modules`` from their shared ``PreTrainedModel`` base,
    which causes ``accelerate`` FSDP2 ``set_auto_wrap_policy`` to raise ``ValueError``.
    """
    present = {type(m).__name__ for m in model.modules()}
    for mod in model.modules():
        no_split = getattr(mod, "_no_split_modules", None)
        if isinstance(no_split, (list, tuple, set)):
            filtered = [name for name in no_split if name in present]
            if len(filtered) != len(no_split):
                mod._no_split_modules = filtered


class QATTrainer(Trainer):
    """HuggingFace ``Trainer`` extended with fake-quantization preparation, static ``amax`` calibration,
    and lossless ``quantizer_state.json`` checkpointing (including FSDP2 + PEFT support).

    Usable standalone for standard SFT / CausalLM Cross-Entropy QAT without any teacher model,
    and composed with :class:`~neural_compressor.torch.distillation.KDTrainer` to form
    :class:`~neural_compressor.torch.qad.QADTrainer`.
    """

    QUANTIZER_STATE_FILENAME = "quantizer_state.json"
    QUANT_CONFIG_FILENAME = "quant_config.json"

    def __init__(
        self,
        model: Union[nn.Module, Any] = None,
        args: Any = None,
        *pos_args,
        quant_config: Optional[FakeQuantConfig] = None,
        calib_batches: int = 0,
        quantizer_state_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        if calib_batches < 0:
            raise ValueError(f"calib_batches must be >= 0, got {calib_batches}")

        if isinstance(model, nn.Module):
            ensure_peft_gradient_checkpointing_delegation(model)
            has_quantizers = any(isinstance(m, TensorQuantizer) for m in model.modules())
            if quant_config is not None and not has_quantizers:
                prepare_qat(model, quant_config)
            if quantizer_state_path is not None:
                load_quantizer_state(model, quantizer_state_path, strict=False)
            _prune_absent_no_split_modules(model)

        resolved_quant_cfg = quant_config
        if resolved_quant_cfg is None and isinstance(model, nn.Module):
            resolved_quant_cfg = getattr(model, "quant_config", None)

        super().__init__(model=model, args=args, *pos_args, **kwargs)
        self.quant_config: Optional[FakeQuantConfig] = resolved_quant_cfg
        self.calib_batches: int = int(calib_batches)
        self._qat_ready: bool = False

    def _uncalibrated_static_quantizers(self) -> list[tuple[str, TensorQuantizer]]:
        base = unwrap_model(self.model)
        return [
            (name, mod)
            for name, mod in base.named_modules()
            if isinstance(mod, TensorQuantizer) and mod.is_enabled and not mod.is_dynamic and mod.amax is None
        ]

    def _build_calib_dataloader(self):
        dataset = self.train_dataset
        if (
            dataset is not None
            and isinstance(dataset, torch.utils.data.Dataset)
            and not isinstance(dataset, torch.utils.data.IterableDataset)
            and hasattr(dataset, "__len__")
        ):
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                sampler: torch.utils.data.Sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, shuffle=False, drop_last=False
                )
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
            )
        return self.get_train_dataloader()

    def _ensure_qat_calibrated_and_finalized(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        *,
        active_model: Optional[nn.Module] = None,
    ) -> None:
        target_model = active_model if active_model is not None else self.model
        base = unwrap_model(target_model)
        if isinstance(resume_from_checkpoint, str):
            state_file = Path(resume_from_checkpoint) / self.QUANTIZER_STATE_FILENAME
            if state_file.is_file():
                load_quantizer_state(base, state_file, strict=False)

        pending = self._uncalibrated_static_quantizers()
        if pending and not self._qat_ready:
            if self.calib_batches > 0:
                logger.info(
                    f"[QATTrainer] Calibrating {len(pending)} static quantizers using {self.calib_batches} batches."
                )
                calibrate_static_amax(
                    target_model,
                    dataloader=self._build_calib_dataloader(),
                    num_batches=self.calib_batches,
                    distributed_sync=True,
                )
            pending = self._uncalibrated_static_quantizers()
            if pending:
                names = [n for n, _ in pending[:5]]
                raise RuntimeError(
                    f"{len(pending)} enabled static TensorQuantizer(s) have no calibrated `amax` ({names}); "
                    "pass `calib_batches > 0`, `quantizer_state_path=...`, or run `calibrate_static_amax()` first."
                )

        for mod in base.modules():
            if isinstance(mod, TensorQuantizer):
                mod.finalize()
        self._qat_ready = True

    def training_step(self, model: nn.Module, inputs: dict[str, Any], *args, **kwargs):
        if not self._qat_ready:
            self._ensure_qat_calibrated_and_finalized(active_model=model)
        return super().training_step(model, inputs, *args, **kwargs)

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, *args, **kwargs):
        if isinstance(resume_from_checkpoint, str):
            state_file = Path(resume_from_checkpoint) / self.QUANTIZER_STATE_FILENAME
            if state_file.is_file():
                load_quantizer_state(unwrap_model(self.model), state_file, strict=False)
        pending = self._uncalibrated_static_quantizers()
        if pending and self.calib_batches > 0 and self.is_fsdp_enabled:
            logger.info(
                f"[QATTrainer] Deferring calibration of {len(pending)} static quantizers until FSDP sharding on GPU."
            )
            return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)
        self._ensure_qat_calibrated_and_finalized(resume_from_checkpoint=resume_from_checkpoint)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)

    def _save_quant_artifacts(self, out_dir: Union[str, Path]) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = unwrap_model(self.model)
        has_static = any(
            isinstance(m, TensorQuantizer) and m.is_enabled and not m.is_dynamic and m.amax is not None
            for m in base.modules()
        )
        if has_static:
            save_quantizer_state(base, out / self.QUANTIZER_STATE_FILENAME)
        if self.quant_config is not None:
            self.quant_config.to_json_file(out / self.QUANT_CONFIG_FILENAME)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        out = output_dir or self.args.output_dir
        if self.is_fsdp_enabled and is_peft_model(self.model):
            if save_fsdp2_peft_adapter(self.model, out, should_save=self.args.should_save):
                if self.args.should_save:
                    os.makedirs(out, exist_ok=True)
                    torch.save(self.args, os.path.join(out, "training_args.bin"))
                    self._save_quant_artifacts(out)
                return
        if self.is_fsdp_enabled and not is_peft_model(self.model):
            dtype = torch.bfloat16 if getattr(self.args, "bf16", False) else None
            if save_fsdp2_full_model(self.model, out, should_save=self.args.should_save, dtype=dtype):
                if self.args.should_save:
                    os.makedirs(out, exist_ok=True)
                    torch.save(self.args, os.path.join(out, "training_args.bin"))
                    self._save_quant_artifacts(out)
                return

        super().save_model(output_dir, _internal_call)
        if self.args.should_save:
            self._save_quant_artifacts(out)


@dataclass
class QuantizationTrainingArguments(TrainingArguments):
    """``transformers.TrainingArguments`` extended with fake-quantization CLI options."""

    quant_recipe: Optional[str] = field(
        default=None,
        metadata={"help": "Optional YAML/JSON FakeQuantConfig path."},
    )
    quant_backend: str = field(
        default="auto",
        metadata={"help": "Fake-quant QDQ backend ('auto', 'triton', or 'pytorch')."},
    )
    calib_batches: int = field(
        default=0,
        metadata={"help": "Number of training batches for automatic static amax calibration."},
    )
    quantizer_state_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to pre-calibrated quantizer_state.json."},
    )
    lora_merge_before_quant: bool = field(
        default=True,
        metadata={"help": "When True (Path-B), merge LoRA delta before weight quantization."},
    )

    def __post_init__(self) -> None:
        fsdp_str = str(getattr(self, "fsdp", "") or "")
        fsdp_cfg = getattr(self, "fsdp_config", None)
        needs_cpu_pg = ("offload" in fsdp_str) or (
            isinstance(fsdp_cfg, dict)
            and (
                bool(fsdp_cfg.get("cpu_offload", False))
                or str(fsdp_cfg.get("state_dict_type", "")) == "FULL_STATE_DICT"
            )
        )
        saved_env: dict[str, Optional[str]] = {}
        if needs_cpu_pg and not dist.is_initialized():
            for key, val in (("ACCELERATE_USE_FSDP", "true"), ("FSDP_OFFLOAD_PARAMS", "true")):
                saved_env[key] = os.environ.get(key)
                if saved_env[key] is None:
                    os.environ[key] = val
        try:
            super().__post_init__()
            if saved_env:
                _ = self.device
        finally:
            for key, prev in saved_env.items():
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev


__all__ = [
    "QATTrainer",
    "QuantizationTrainingArguments",
    "ensure_peft_gradient_checkpointing_delegation",
    "is_peft_model",
    "save_fsdp2_full_model",
    "save_fsdp2_peft_adapter",
    "unwrap_model",
]
