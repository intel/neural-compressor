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
"""QADTrainer composing KDTrainer + QATTrainer via cooperative multiple inheritance."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from neural_compressor.torch.distillation import DistillationConfig, KDTrainer
from neural_compressor.torch.quantization.fake_quant import (
    FakeQuantConfig,
    TensorQuantizer,
    convert_peft_lora_to_quant_lora,
    ensure_peft_torchao_compat,
    register_peft_quant_modules,
)
from neural_compressor.torch.quantization.qat_trainer import (
    QATTrainer,
    ensure_peft_gradient_checkpointing_delegation,
    is_peft_model,
    unwrap_model,
)
from neural_compressor.torch.quantization.quantize import prepare_qat

from .args import QADTrainingArguments
from .metrics import device_hardware_name, memory_report, param_counts, reset_peak_memory, tflops_and_mfu


def prepare_qad_model(
    model: nn.Module,
    quant_config: FakeQuantConfig,
    *,
    lora_config: Optional[Any] = None,
    lora_merge_before_quant: bool = True,
) -> nn.Module:
    """Prepare ``model`` for QAD by inserting fake-quantizers and optionally attaching PEFT LoRA.

    Args:
        model: Standard ``transformers.PreTrainedModel`` (or ``PeftModel``).
        quant_config: :class:`FakeQuantConfig` describing weight and activation fake-quantization.
        lora_config: Optional ``peft.LoraConfig`` instance or ``dict`` of ``LoraConfig`` kwargs.
        lora_merge_before_quant: When ``True`` (Path-B, recommended for inference export parity),
            quantizes ``W + s * B @ A`` before weight fake-quantization.
    """
    if not any(isinstance(m, TensorQuantizer) for m in model.modules()):
        prepare_qat(model, quant_config)
    if lora_config is not None:
        from peft import LoraConfig, get_peft_model

        ensure_peft_torchao_compat()
        register_peft_quant_modules()
        peft_cfg = lora_config if isinstance(lora_config, LoraConfig) else LoraConfig(**dict(lora_config))
        model = get_peft_model(model, peft_cfg)
        quant_order = "lora_then_quant" if lora_merge_before_quant else "quant_then_lora"
        convert_peft_lora_to_quant_lora(model, quant_order=quant_order)
        ensure_peft_gradient_checkpointing_delegation(model)
    return model


class QADTrainer(KDTrainer, QATTrainer):
    """Unified Quantization-Aware Distillation Trainer composing :class:`KDTrainer` and :class:`QATTrainer`.

    MRO: ``QADTrainer -> KDTrainer -> QATTrainer -> transformers.Trainer``
    * ``KDTrainer.compute_loss`` provides chunked zero-logit-allocation Top-K KL / Full KL / Hybrid CE
      across online external teacher, zero-VRAM LoRA ``teacher_model="self"``, and offline Top-K shards.
    * ``QATTrainer`` provides ``prepare_qat``, automatic static ``amax`` calibration, FSDP2 + PEFT
      checkpoint gathering, and lossless ``quantizer_state.json`` + ``quant_config.json`` persistence.
    """

    def __init__(
        self,
        model: Union[nn.Module, Any] = None,
        args: Any = None,
        *pos_args,
        teacher_model: Optional[Union[nn.Module, str]] = None,
        teacher: Optional[Union[nn.Module, str]] = None,
        distill_config: Optional[DistillationConfig] = None,
        quant_config: Optional[FakeQuantConfig] = None,
        calib_batches: Optional[int] = None,
        quantizer_state_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        if distill_config is None and isinstance(args, QADTrainingArguments):
            distill_config = args.build_distill_config()
        if distill_config is None:
            raise ValueError("QADTrainer requires `distill_config` or `QADTrainingArguments`.")

        if quant_config is None and isinstance(args, QADTrainingArguments):
            quant_config = args.build_quant_config()
        if calib_batches is None:
            calib_batches = int(getattr(args, "calib_batches", 0) or 0)
        if quantizer_state_path is None:
            quantizer_state_path = getattr(args, "quantizer_state_path", None)

        self._last_log_time = time.time()
        self._last_log_step = 0
        self._tokens_since_last_log = 0
        self._peak_reset = False

        super().__init__(
            model=model,
            args=args,
            *pos_args,
            teacher_model=teacher_model,
            teacher=teacher,
            distill_config=distill_config,
            quant_config=quant_config,
            calib_batches=calib_batches,
            quantizer_state_path=quantizer_state_path,
            **kwargs,
        )
        self.mode = "lora" if (model is not None and is_peft_model(unwrap_model(self.model))) else "full"

    def compute_loss(self, model: nn.Module, inputs: dict[str, Any], return_outputs: bool = False, **kwargs):
        if model.training and "input_ids" in inputs:
            self._tokens_since_last_log += int(inputs["input_ids"].numel())
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def training_step(self, *args, **kwargs):
        if not self._peak_reset:
            reset_peak_memory(self.args.device)
            self._last_log_time = time.time()
            self._peak_reset = True
        return super().training_step(*args, **kwargs)

    def log(self, logs: dict[str, Any], start_time: Optional[float] = None) -> None:
        now = time.time()
        step = int(self.state.global_step)
        delta = step - self._last_log_step
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        device = self.args.device
        device_name = device_hardware_name(device)
        if delta > 0:
            if self._tokens_since_last_log > 0:
                tokens = self._tokens_since_last_log * world
            else:
                seq = int(getattr(self.args, "seq_length", 0) or 0)
                tokens = (
                    delta * self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * world * seq
                )
            seconds = max(now - self._last_log_time, 1e-9)
            counts = param_counts(unwrap_model(self.model))
            if self.self_teacher:
                teacher_params = max(counts["params_total"] - counts["params_trainable"], 0)
            elif self.teacher is not None:
                teacher_params = param_counts(unwrap_model(self.teacher))["params_total"]
            else:
                teacher_params = 0
            perf = tflops_and_mfu(
                tokens=tokens,
                step_seconds=seconds,
                params=counts["params_total"],
                mode=self.mode,
                teacher_params=teacher_params,
                device_name=device_name,
                world_size=world,
                recompute=bool(
                    self.args.gradient_checkpointing
                    or (self.args.fsdp_config or {}).get("activation_checkpointing", False)
                ),
            )
            logs.update(
                {
                    "tokens": float(tokens),
                    "tokens_per_s": tokens / seconds,
                    "step_s": seconds / delta,
                    **counts,
                    **{key: value for key, value in perf.items() if value is not None},
                }
            )
            self._tokens_since_last_log = 0
        memory = memory_report(device)
        logs["peak_gib"] = memory["peak"] / (1024**3)
        logs["alloc_gib"] = memory["allocated"] / (1024**3)
        logs.update(self._last_loss_parts)
        self._last_log_time, self._last_log_step = now, step
        super().log(logs, start_time=start_time)


__all__ = ["QADTrainer", "prepare_qad_model"]
