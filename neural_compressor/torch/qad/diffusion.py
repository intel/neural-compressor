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
"""Quantization-Aware Distillation (QAD) for Diffusion Transformers (DiT) and UNet models.

Supports Flow-Matching (Wan2.1/Wan2.2, Flux, SD3, HunyuanVideo, Qwen-Image) and DDPM/v-prediction
denoising architectures with:
* Zero-extra-VRAM Self-Teacher LoRA Path-B (``teacher_model="self"``) and external/sharded teachers.
* Velocity / noise prediction alignment (``mse``, ``huber``, ``cosine_mse``) in FP32.
* Optional ground-truth flow-matching / denoising task supervision (``task_weight``).
* Optional intermediate transformer-block feature alignment (``feature_kd_weight``).
* Automatic multi-timestep static FP8 ``amax`` calibration across the DiT attention projections.
"""

from __future__ import annotations

import math
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from neural_compressor.torch.algorithms.fake_quant import (
    FakeQuantConfig,
    TensorQuantizer,
    calibrate_static_amax,
    convert_peft_lora_to_quant_lora,
    ensure_peft_torchao_compat,
    register_peft_quant_modules,
)
from neural_compressor.torch.algorithms.fake_quant.trainer import (
    QATTrainer,
    QuantizationTrainingArguments,
    is_peft_model,
    unwrap_model,
)
from neural_compressor.torch.distillation.trainer import _place_teacher_model, _self_teacher_context
from neural_compressor.torch.quantization.quantize import prepare_qat

from .utils import _find_layers, memory_report, param_counts, reset_peak_memory

DiffusionLossType = Literal["mse", "huber", "cosine_mse"]
DiffusionPredictionType = Literal["flow_matching", "epsilon", "v_prediction"]
TimestepSamplingType = Literal["logit_normal", "uniform"]


@dataclass
class DiffusionQADConfig:
    """Configuration for Diffusion / Flow-Matching Quantization-Aware Distillation."""

    teacher: Optional[Union[nn.Module, str]] = "self"
    teacher_device: str = "self"
    loss_type: DiffusionLossType = "mse"
    huber_delta: float = 0.1
    kd_weight: float = 1.0
    task_weight: float = 0.0
    feature_kd_weight: float = 0.0
    feature_block_indices: Optional[list[int]] = None
    prediction_type: DiffusionPredictionType = "flow_matching"
    num_train_timesteps: int = 1000
    flow_shift: float = 5.0
    timestep_sampling: TimestepSamplingType = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    latent_key: str = "latents"
    noisy_latent_key: str = "hidden_states"
    timestep_key: str = "timestep"

    def __post_init__(self) -> None:
        if self.loss_type not in ("mse", "huber", "cosine_mse"):
            raise ValueError(f"Unsupported loss_type {self.loss_type!r}")
        if self.prediction_type not in ("flow_matching", "epsilon", "v_prediction"):
            raise ValueError(f"Unsupported prediction_type {self.prediction_type!r}")
        if self.timestep_sampling not in ("logit_normal", "uniform"):
            raise ValueError(f"Unsupported timestep_sampling {self.timestep_sampling!r}")
        if self.kd_weight < 0 or self.task_weight < 0 or self.feature_kd_weight < 0:
            raise ValueError("kd_weight, task_weight, and feature_kd_weight must be >= 0")
        if self.kd_weight == 0 and self.task_weight == 0 and self.feature_kd_weight == 0:
            raise ValueError("At least one of kd_weight, task_weight, or feature_kd_weight must be > 0")
        if self.num_train_timesteps <= 0:
            raise ValueError(f"num_train_timesteps must be > 0, got {self.num_train_timesteps}")
        if self.flow_shift <= 0:
            raise ValueError(f"flow_shift must be > 0, got {self.flow_shift}")


@dataclass
class DiffusionQADTrainingArguments(QuantizationTrainingArguments):
    """HuggingFace ``TrainingArguments`` extended with Diffusion QAD parameters."""

    teacher_model_name_or_path: Optional[str] = field(
        default="self",
        metadata={"help": "Teacher DiT/UNet path or 'self' for zero-extra-VRAM LoRA self-distillation."},
    )
    teacher_placement: str = field(
        default="self",
        metadata={"help": "Teacher placement ('self', 'shard', 'cuda', or 'cpu')."},
    )
    diffusion_loss_type: DiffusionLossType = field(
        default="mse",
        metadata={"help": "Prediction distillation loss ('mse', 'huber', or 'cosine_mse')."},
    )
    diffusion_kd_weight: float = field(default=1.0, metadata={"help": "Weight of teacher prediction alignment loss."})
    diffusion_task_weight: float = field(default=0.0, metadata={"help": "Weight of ground-truth velocity/noise loss."})
    diffusion_feature_kd_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of intermediate transformer block feature alignment loss."},
    )
    prediction_type: DiffusionPredictionType = field(
        default="flow_matching",
        metadata={"help": "Denoising parameterization ('flow_matching', 'epsilon', or 'v_prediction')."},
    )
    num_train_timesteps: int = field(default=1000, metadata={"help": "Scheduler training timestep horizon."})
    flow_shift: float = field(
        default=5.0, metadata={"help": "Flow-matching sigma shift (5.0 for Wan, 3.0 for SD3/Flux)."}
    )
    timestep_sampling: TimestepSamplingType = field(
        default="logit_normal",
        metadata={"help": "Timestep distribution ('logit_normal' or 'uniform')."},
    )

    def build_diffusion_qad_config(self) -> DiffusionQADConfig:
        return DiffusionQADConfig(
            teacher=self.teacher_model_name_or_path,
            teacher_device=self.teacher_placement,
            loss_type=self.diffusion_loss_type,
            kd_weight=self.diffusion_kd_weight,
            task_weight=self.diffusion_task_weight,
            feature_kd_weight=self.diffusion_feature_kd_weight,
            prediction_type=self.prediction_type,
            num_train_timesteps=self.num_train_timesteps,
            flow_shift=self.flow_shift,
            timestep_sampling=self.timestep_sampling,
        )


def _ensure_diffusion_peft_grad_checkpointing(model: nn.Module) -> None:
    """Delegate gradient checkpointing on Diffusers ``PeftModel`` and mark stem outputs ``requires_grad``."""
    inner = unwrap_model(model)
    core = getattr(getattr(inner, "base_model", None), "model", inner)

    if inner is not core and not hasattr(inner, "_nc_diff_gc_patched"):

        def _gc_enable(**kwargs):
            if hasattr(core, "enable_gradient_checkpointing"):
                return core.enable_gradient_checkpointing()
            if hasattr(core, "gradient_checkpointing_enable"):
                return core.gradient_checkpointing_enable(**kwargs)
            return None

        def _gc_disable():
            if hasattr(core, "disable_gradient_checkpointing"):
                return core.disable_gradient_checkpointing()
            if hasattr(core, "gradient_checkpointing_disable"):
                return core.gradient_checkpointing_disable()
            return None

        inner.gradient_checkpointing_enable = _gc_enable  # type: ignore[method-assign]
        inner.gradient_checkpointing_disable = _gc_disable  # type: ignore[method-assign]
        object.__setattr__(inner, "_nc_diff_gc_patched", True)

    stem_candidates = ("patch_embedding", "pos_embed", "x_embedder", "proj_in", "conv_in")
    for attr in stem_candidates:
        stem = getattr(core, attr, None)
        if isinstance(stem, nn.Module) and not getattr(stem, "_nc_diff_stem_grad_hooked", False):

            def _require_grad_hook(_mod: nn.Module, _inp: Any, output: Any) -> Any:
                if not torch.is_grad_enabled():
                    return output
                if isinstance(output, torch.Tensor):
                    output.requires_grad_(True)
                elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                    output[0].requires_grad_(True)
                return output

            stem.register_forward_hook(_require_grad_hook)
            object.__setattr__(stem, "_nc_diff_stem_grad_hooked", True)


def prepare_diffusion_qad_model(
    model: nn.Module,
    quant_config: FakeQuantConfig,
    *,
    lora_config: Optional[Any] = None,
    lora_merge_before_quant: bool = True,
) -> nn.Module:
    """Insert fake-quantizers into a Diffusers DiT/UNet model and optionally attach Path-B LoRA."""
    if not any(isinstance(m, TensorQuantizer) for m in model.modules()):
        prepare_qat(model, quant_config)
    if lora_config is not None:
        from peft import LoraConfig, get_peft_model

        ensure_peft_torchao_compat()
        register_peft_quant_modules()
        if isinstance(lora_config, LoraConfig):
            peft_cfg = lora_config
        else:
            cfg_dict = dict(lora_config)
            cfg_dict.pop("task_type", None)
            peft_cfg = LoraConfig(**cfg_dict)
        model = get_peft_model(model, peft_cfg)
        quant_order = "lora_then_quant" if lora_merge_before_quant else "quant_then_lora"
        convert_peft_lora_to_quant_lora(model, quant_order=quant_order)
        _ensure_diffusion_peft_grad_checkpointing(model)
    return model


def _extract_prediction_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "sample") and isinstance(output.sample, torch.Tensor):
        return output.sample
    if isinstance(output, Mapping) and isinstance(output.get("sample"), torch.Tensor):
        return output["sample"]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Cannot extract prediction tensor from model output of type {type(output).__name__}")


def _prediction_alignment_loss(
    student_pred: torch.Tensor,
    target_pred: torch.Tensor,
    *,
    loss_type: DiffusionLossType = "mse",
    huber_delta: float = 0.1,
) -> torch.Tensor:
    s_f32 = student_pred.float()
    t_f32 = target_pred.float()
    if loss_type == "mse":
        return F.mse_loss(s_f32, t_f32)
    if loss_type == "huber":
        return F.huber_loss(s_f32, t_f32, delta=huber_delta)
    if loss_type == "cosine_mse":
        mse = F.mse_loss(s_f32, t_f32)
        s_flat = s_f32.reshape(s_f32.size(0), -1)
        t_flat = t_f32.reshape(t_f32.size(0), -1)
        cos = (1.0 - F.cosine_similarity(s_flat, t_flat, dim=-1, eps=1e-8)).mean()
        return mse + 0.1 * cos
    raise ValueError(f"Unknown loss_type {loss_type!r}")


@contextmanager
def _capture_block_features(
    model: nn.Module,
    block_indices: Sequence[int],
    *,
    detach: bool = False,
) -> Iterator[list[torch.Tensor]]:
    captured: list[torch.Tensor] = []
    layers = _find_layers(unwrap_model(model))
    if layers is None or not block_indices:
        yield captured
        return

    handles = []
    n_layers = len(layers)
    for idx in block_indices:
        resolved_idx = idx if idx >= 0 else (n_layers + idx)
        if 0 <= resolved_idx < n_layers:

            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                feat = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(feat, torch.Tensor):
                    captured.append(feat.detach() if detach else feat)

            handles.append(layers[resolved_idx].register_forward_hook(_hook))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


class DiffusionQADTrainer(QATTrainer):
    """Quantization-Aware Distillation Trainer for Diffusion / Flow-Matching DiTs and UNets."""

    def __init__(
        self,
        model: Union[nn.Module, Any] = None,
        args: Any = None,
        *pos_args,
        teacher_model: Optional[Union[nn.Module, str]] = None,
        teacher: Optional[Union[nn.Module, str]] = None,
        diffusion_config: Optional[DiffusionQADConfig] = None,
        quant_config: Optional[FakeQuantConfig] = None,
        calib_batches: Optional[int] = None,
        quantizer_state_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        resolved_teacher = teacher_model if teacher_model is not None else teacher
        if diffusion_config is None:
            if isinstance(args, DiffusionQADTrainingArguments):
                diffusion_config = args.build_diffusion_qad_config()
            else:
                diffusion_config = DiffusionQADConfig(
                    teacher=resolved_teacher if resolved_teacher is not None else "self"
                )
        if resolved_teacher is None and diffusion_config.teacher is not None:
            resolved_teacher = diffusion_config.teacher

        self.diffusion_config: DiffusionQADConfig = diffusion_config
        self.self_teacher: bool = resolved_teacher == "self" or diffusion_config.teacher_device == "self"

        if self.self_teacher:
            self.teacher: Optional[nn.Module] = None
        elif isinstance(resolved_teacher, nn.Module):
            self.teacher = resolved_teacher
            self.teacher.eval()
            self.teacher.requires_grad_(False)
        else:
            self.teacher = None

        if quant_config is None and hasattr(args, "build_quant_config") and callable(args.build_quant_config):
            quant_config = args.build_quant_config()
        if calib_batches is None:
            calib_batches = int(getattr(args, "calib_batches", 0) or 0)
        if quantizer_state_path is None:
            quantizer_state_path = getattr(args, "quantizer_state_path", None)

        if isinstance(model, nn.Module):
            _ensure_diffusion_peft_grad_checkpointing(model)

        self._last_loss_parts: dict[str, float] = {}
        self._last_log_time = time.time()
        self._last_log_step = 0
        self._peak_reset = False

        super().__init__(
            model=model,
            args=args,
            *pos_args,
            quant_config=quant_config,
            calib_batches=calib_batches,
            quantizer_state_path=quantizer_state_path,
            **kwargs,
        )
        self.model_accepts_loss_kwargs = False
        if self.teacher is not None:
            self.teacher = _place_teacher_model(self.teacher, self.diffusion_config.teacher_device, self.args.device)
        self.mode = "lora" if (model is not None and is_peft_model(unwrap_model(self.model))) else "full"

    # ------------------------------------------------------------------
    # Flow-Matching / Denoising Trajectory Construction
    # ------------------------------------------------------------------

    def _sample_sigmas(
        self, batch_size: int, device: torch.device, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        cfg = self.diffusion_config
        if cfg.timestep_sampling == "logit_normal":
            normal = torch.randn((batch_size,), device=device, dtype=torch.float32, generator=generator)
            u = torch.sigmoid(cfg.logit_mean + cfg.logit_std * normal)
        else:
            u = torch.rand((batch_size,), device=device, dtype=torch.float32, generator=generator)
        u = u.clamp(1e-4, 1.0 - 1e-4)
        if cfg.flow_shift != 1.0:
            s = cfg.flow_shift
            u = (s * u) / (1.0 + (s - 1.0) * u)
        return u

    def _prepare_forward_kwargs_and_target(
        self,
        inputs: Mapping[str, Any],
        device: torch.device,
    ) -> tuple[dict[str, Any], Optional[torch.Tensor]]:
        cfg = self.diffusion_config
        reserved = {cfg.latent_key, "noise", "target"}
        extra_kwargs: dict[str, Any] = {}
        for key, val in inputs.items():
            if key in reserved:
                continue
            extra_kwargs[key] = val.to(device) if isinstance(val, torch.Tensor) else val

        if cfg.noisy_latent_key in extra_kwargs and cfg.timestep_key in extra_kwargs:
            target = inputs.get("target")
            if isinstance(target, torch.Tensor):
                target = target.to(device)
            extra_kwargs.setdefault("return_dict", False)
            return extra_kwargs, target

        if "sample" in extra_kwargs and cfg.timestep_key in extra_kwargs:
            target = inputs.get("target")
            if isinstance(target, torch.Tensor):
                target = target.to(device)
            extra_kwargs.setdefault("return_dict", False)
            return extra_kwargs, target

        if cfg.latent_key not in inputs:
            raise KeyError(
                f"DiffusionQADTrainer expects either '{cfg.latent_key}' or "
                f"('{cfg.noisy_latent_key}', '{cfg.timestep_key}') in batch, got keys {sorted(inputs.keys())}."
            )

        clean_latents = inputs[cfg.latent_key].to(device)
        bsz = clean_latents.size(0)
        noise = (
            inputs["noise"].to(device)
            if isinstance(inputs.get("noise"), torch.Tensor)
            else torch.randn_like(clean_latents)
        )
        sigmas = self._sample_sigmas(bsz, device)
        expand_shape = (bsz,) + (1,) * (clean_latents.ndim - 1)
        sigma_b = sigmas.view(expand_shape)

        if cfg.prediction_type == "flow_matching":
            noisy = ((1.0 - sigma_b) * clean_latents.float() + sigma_b * noise.float()).to(clean_latents.dtype)
            target = (noise.float() - clean_latents.float()).to(clean_latents.dtype)
            timesteps = sigmas * float(cfg.num_train_timesteps)
        else:
            # Cosine / linear DDPM interpolation fallback for epsilon / v_prediction UNets & DiTs
            angles = sigmas * (0.5 * math.pi)
            alpha_b = torch.cos(angles).view(expand_shape)
            std_b = torch.sin(angles).view(expand_shape)
            noisy = (alpha_b * clean_latents.float() + std_b * noise.float()).to(clean_latents.dtype)
            if cfg.prediction_type == "v_prediction":
                target = (alpha_b * noise.float() - std_b * clean_latents.float()).to(clean_latents.dtype)
            else:
                target = noise
            timesteps = (sigmas * float(cfg.num_train_timesteps - 1)).round().long()

        extra_kwargs[cfg.noisy_latent_key] = noisy
        extra_kwargs[cfg.timestep_key] = timesteps
        extra_kwargs.setdefault("return_dict", False)
        return extra_kwargs, target

    # ------------------------------------------------------------------
    # Calibration & Distillation Loss
    # ------------------------------------------------------------------

    def _run_static_calibration(self, target_model: nn.Module) -> None:
        dl = self._build_calib_dataloader()

        def _diffusion_calib_fn(m: nn.Module) -> None:
            for step, batch in enumerate(dl):
                if step >= self.calib_batches:
                    break
                fwd_kwargs, _ = self._prepare_forward_kwargs_and_target(batch, self.args.device)
                m(**fwd_kwargs)

        calibrate_static_amax(
            target_model,
            calib_fn=_diffusion_calib_fn,
            distributed_sync=True,
            device=self.args.device,
        )

    def training_step(self, *args, **kwargs):
        if not self._peak_reset:
            reset_peak_memory(self.args.device)
            self._last_log_time = time.time()
            self._peak_reset = True
        return super().training_step(*args, **kwargs)

    def compute_loss(self, model: nn.Module, inputs: dict[str, Any], return_outputs: bool = False, **kwargs):
        cfg = self.diffusion_config
        fwd_kwargs, gt_target = self._prepare_forward_kwargs_and_target(inputs, self.args.device)

        need_teacher = cfg.kd_weight > 0.0 or cfg.feature_kd_weight > 0.0
        block_indices = cfg.feature_block_indices or []
        need_feat = cfg.feature_kd_weight > 0.0 and len(block_indices) > 0

        teacher_pred: Optional[torch.Tensor] = None
        teacher_feats: list[torch.Tensor] = []

        if need_teacher:
            if self.self_teacher:
                with ExitStack() as stack:
                    stack.enter_context(_self_teacher_context(model))
                    if need_feat:
                        teacher_feats = stack.enter_context(_capture_block_features(model, block_indices, detach=True))
                    t_out = model(**fwd_kwargs)
                    teacher_pred = _extract_prediction_tensor(t_out).detach()
            elif self.teacher is not None:
                with ExitStack() as stack:
                    stack.enter_context(torch.no_grad())
                    if need_feat:
                        teacher_feats = stack.enter_context(
                            _capture_block_features(self.teacher, block_indices, detach=True)
                        )
                    t_out = self.teacher(**fwd_kwargs)
                    teacher_pred = _extract_prediction_tensor(t_out).detach()
            else:
                raise ValueError(
                    "DiffusionQADTrainer requires `teacher_model='self'` or an `nn.Module` teacher when kd_weight > 0."
                )

        with ExitStack() as stack:
            student_feats: list[torch.Tensor] = []
            if need_feat:
                student_feats = stack.enter_context(_capture_block_features(model, block_indices, detach=False))
            s_out = model(**fwd_kwargs)
            student_pred = _extract_prediction_tensor(s_out)

        total_loss = student_pred.float().sum() * 0.0
        parts: dict[str, float] = {}

        if cfg.kd_weight > 0.0 and teacher_pred is not None:
            kd_loss = _prediction_alignment_loss(
                student_pred,
                teacher_pred,
                loss_type=cfg.loss_type,
                huber_delta=cfg.huber_delta,
            )
            total_loss = total_loss + cfg.kd_weight * kd_loss
            parts["kd_loss"] = float(kd_loss.detach().item())

        if cfg.task_weight > 0.0:
            if gt_target is None:
                raise ValueError("diffusion task_weight > 0 requires clean `latents` or explicit `target` in batch.")
            task_loss = _prediction_alignment_loss(
                student_pred,
                gt_target,
                loss_type=cfg.loss_type,
                huber_delta=cfg.huber_delta,
            )
            total_loss = total_loss + cfg.task_weight * task_loss
            parts["task_loss"] = float(task_loss.detach().item())

        if need_feat and student_feats and len(student_feats) == len(teacher_feats):
            feat_terms = []
            for s_f, t_f in zip(student_feats, teacher_feats):
                s_32 = s_f.float()
                t_32 = t_f.float()
                denom = t_32.pow(2).mean().clamp_min(1e-6)
                feat_terms.append(F.mse_loss(s_32, t_32) / denom)
            feat_loss = torch.stack(feat_terms).mean()
            total_loss = total_loss + cfg.feature_kd_weight * feat_loss
            parts["feat_loss"] = float(feat_loss.detach().item())

        self._last_loss_parts = parts
        return (total_loss, s_out) if return_outputs else total_loss

    def log(self, logs: dict[str, Any], start_time: Optional[float] = None) -> None:
        now = time.time()
        step = int(self.state.global_step)
        delta = step - self._last_log_step
        if delta > 0:
            seconds = max(now - self._last_log_time, 1e-9)
            counts = param_counts(unwrap_model(self.model))
            logs.update(
                {
                    "step_s": seconds / delta,
                    **counts,
                }
            )
        memory = memory_report(self.args.device)
        logs["peak_gib"] = memory["peak"] / (1024**3)
        logs["alloc_gib"] = memory["allocated"] / (1024**3)
        logs.update(self._last_loss_parts)
        self._last_log_time, self._last_log_step = now, step
        super().log(logs, start_time=start_time)


__all__ = [
    "DiffusionLossType",
    "DiffusionPredictionType",
    "DiffusionQADConfig",
    "DiffusionQADTrainer",
    "DiffusionQADTrainingArguments",
    "TimestepSamplingType",
    "prepare_diffusion_qad_model",
]
