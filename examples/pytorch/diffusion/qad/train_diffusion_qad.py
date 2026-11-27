#!/usr/bin/env python3
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
"""Diffusion Transformer (DiT) Quantization-Aware Distillation (QAD) entrypoint.

Supports Wan2.1 / Wan2.2 (``WanTransformer3DModel``) and generic Diffusers DiT models
with MXFP4+FP8 hybrid fake-quantization, Path-B Self-Teacher LoRA or Full-Parameters FSDP2,
optional UMT5/T5 real prompt encoding, and static FP8 ``amax`` calibration.

Examples:
    # 4-GPU MXFP4+FP8 LoRA Path-B QAD on Wan2.2-TI2V-5B-Diffusers:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
        examples/pytorch/diffusion/qad/train_diffusion_qad.py \\
        --config examples/pytorch/diffusion/qad/recipes/wan2_2_ti2v_5b_mxfp4_fp8_lora_path_b.yaml
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.utils.data import Dataset
from transformers import HfArgumentParser

from neural_compressor.torch.algorithms.fake_quant import FakeQuantConfig, save_quantizer_state
from neural_compressor.torch.algorithms.fake_quant.trainer import is_peft_model, unwrap_model
from neural_compressor.torch.qad import (
    DiffusionQADConfig,
    DiffusionQADTrainer,
    DiffusionQADTrainingArguments,
    GradDebugCallback,
    StopAfterSteps,
    decoder_layer_class_names,
    prepare_diffusion_qad_model,
)

DEFAULT_VIDEO_PROMPTS: list[str] = [
    "A cinematic aerial tracking shot over snow-covered alpine peaks at sunrise, golden volumetric light, crisp 4K detail.",
    "Close-up macro slow-motion shot of raindrops rippling across a neon-lit puddle in Tokyo at night, shallow depth of field.",
    "An astronaut walking across basaltic ridges on Mars as dust swirls in low gravity, sunlight glinting off the visor.",
    "Time-lapse of bioluminescent waves washing onto a black-sand beach under the Milky Way galaxy, vivid turquoise glow.",
    "A red steam locomotive crossing a stone viaduct above an autumn valley shrouded in morning mist, filmic color grading.",
    "Underwater tracking shot of a sea turtle gliding over a colorful coral reef with sunbeams piercing clear blue water.",
    "A cozy artisan pottery studio where hands shape wet clay on a spinning wheel, warm window light, floating dust motes.",
    "Wide architectural shot of futuristic glass botanical domes in a misty mountain forest, waterfalls cascading inside.",
    "A hummingbird hovering beside scarlet hibiscus blossoms in ultra-slow motion, iridescent feathers sharp in natural light.",
    "Drone shot following a sailboat cutting through deep sapphire ocean swells at sunset, spray catching warm orange light.",
    "A bustling traditional lantern market along a canal at dusk, hundreds of glowing paper lanterns reflected on the water.",
    "Close-up of molten glass being blown and twisted in a furnace workshop, radiant amber glow and crisp specular highlights.",
    "A snow leopard stepping silently across frost-covered granite boulders in the Himalayas, breath visible in cold air.",
    "Top-down shot of emerald river braids winding through volcanic ash plains in Iceland, dramatic cloud shadows moving.",
    "A classical violinist performing on an empty wooden concert stage under a single dramatic spotlight, expressive motion.",
    " sunlit redwood forest cathedral with towering trunks and drifting fog rays, slow dolly forward along a mossy trail.",
]


@dataclass
class DiffusionModelDataArguments:
    """Model, latent shape, and conditioning arguments for Diffusion QAD."""

    model_name_or_path: str = field(
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        metadata={"help": "Diffusers pipeline directory / Hub ID or direct transformer directory."},
    )
    subfolder: Optional[str] = field(
        default="transformer",
        metadata={"help": "Subfolder containing the DiT model inside a Diffusers pipeline repo (or None)."},
    )
    tiny: bool = field(
        default=False,
        metadata={"help": "Instantiate a tiny randomly initialized WanTransformer3DModel for fast testing."},
    )
    model_dtype: str = field(default="bfloat16", metadata={"help": "Model load dtype (bfloat16/float16/float32)."})
    latents_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional directory of pre-saved `.pt` dicts containing `latents` and `encoder_hidden_states`."
        },
    )
    encode_real_prompts: bool = field(
        default=True,
        metadata={"help": "Encode real text prompts with the pipeline's text_encoder before loading the DiT."},
    )
    num_samples: int = field(
        default=128,
        metadata={"help": "Number of latent-prompt training samples when `latents_dir` is not provided."},
    )
    latent_frames: int = field(default=5, metadata={"help": "Latent temporal frames F (patch_size_t=1 -> F tokens)."})
    latent_height: int = field(default=16, metadata={"help": "Latent height H (patch_size_h=2 -> H/2 tokens)."})
    latent_width: int = field(default=16, metadata={"help": "Latent width W (patch_size_w=2 -> W/2 tokens)."})
    max_text_seq_len: int = field(default=128, metadata={"help": "Text conditioning token length L_txt."})
    ti2v_zero_first_frame_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of TI2V first-frame conditioning (frame-0 t=0, frames 1..F-1 t=sigma*1000)."},
    )
    use_lora: bool = field(default=False, metadata={"help": "Attach Path-B QuantLoraLinear adapters to DiT."})
    lora_r: int = field(default=32, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout."})
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Explicit LoRA target module suffixes for DiT blocks."},
    )
    feature_block_indices: Optional[list[int]] = field(
        default=None,
        metadata={"help": "Optional transformer block indices for intermediate feature MSE distillation."},
    )
    grad_debug: bool = field(default=False, metadata={"help": "Check finite gradients before optimizer steps."})
    stop_after_steps: int = field(default=0, metadata={"help": "Optional early stop step count."})


class LatentPromptDataset(Dataset):
    """Dataset of clean spatio-temporal latents paired with text conditioning embeddings."""

    def __init__(
        self,
        *,
        num_samples: int,
        in_channels: int,
        num_frames: int,
        height: int,
        width: int,
        prompt_embeds_bank: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.prompt_embeds_bank = prompt_embeds_bank.cpu().to(dtype)
        self.dtype = dtype
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator(device="cpu").manual_seed(self.seed + int(idx) * 1009)
        # Smooth + detail multiscale spatio-temporal latent field mimicking VAE video latents
        low_freq = torch.randn(
            (self.in_channels, max(1, self.num_frames), max(2, self.height // 2), max(2, self.width // 2)),
            generator=gen,
            dtype=torch.float32,
        )
        low_up = torch.nn.functional.interpolate(
            low_freq, size=(self.height, self.width), mode="bilinear", align_corners=False
        )
        detail = 0.35 * torch.randn(
            (self.in_channels, self.num_frames, self.height, self.width),
            generator=gen,
            dtype=torch.float32,
        )
        latents = ((low_up + detail) / 1.06).to(self.dtype)
        prompt_embed = self.prompt_embeds_bank[idx % self.prompt_embeds_bank.size(0)].clone()
        return {
            "latents": latents,
            "encoder_hidden_states": prompt_embed,
        }


class DirectoryLatentDataset(Dataset):
    """Loads pre-saved `.pt` latent + prompt embedding dicts from disk."""

    def __init__(self, latents_dir: str, dtype: torch.dtype = torch.bfloat16) -> None:
        self.files = sorted(Path(latents_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No `.pt` files found in {latents_dir!r}")
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        out: dict[str, torch.Tensor] = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                out[k] = v.to(self.dtype)
            elif isinstance(v, torch.Tensor):
                out[k] = v
        return out


def _collate_diffusion_batch(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = features[0].keys()
    return {k: torch.stack([f[k] for f in features], dim=0) for k in keys}


def _encode_prompt_bank(
    pipeline_path: str,
    *,
    prompts: list[str],
    max_seq_len: int,
    text_dim: int,
    dtype: torch.dtype,
    enable_real_encoder: bool,
    local_rank: int,
    world_size: int,
) -> torch.Tensor:
    """Encode ``prompts`` once with pipeline ``tokenizer`` + ``text_encoder`` and free VRAM immediately."""
    root = Path(pipeline_path)
    has_text_encoder = (
        enable_real_encoder and root.is_dir() and (root / "text_encoder").is_dir() and (root / "tokenizer").is_dir()
    )

    if not has_text_encoder:
        gen = torch.Generator(device="cpu").manual_seed(1234)
        return torch.randn((len(prompts), max_seq_len, text_dim), generator=gen, dtype=dtype)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    bank_cpu: Optional[torch.Tensor] = None

    if local_rank == 0 or not dist.is_initialized():
        from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel, UMT5EncoderModel

        tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
        enc_cfg = AutoConfig.from_pretrained(str(root / "text_encoder"))
        enc_cls = {
            "umt5": UMT5EncoderModel,
            "t5": T5EncoderModel,
        }.get(getattr(enc_cfg, "model_type", ""), AutoModel)
        txt_enc = enc_cls.from_pretrained(
            str(root / "text_encoder"),
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        txt_enc.eval()
        enc_runner = (
            txt_enc.get_encoder() if hasattr(txt_enc, "get_encoder") and callable(txt_enc.get_encoder) else txt_enc
        )
        encoded = tok(
            prompts,
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        attn_mask = encoded.attention_mask.to(device)
        with torch.no_grad():
            embeds = enc_runner(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            embeds = embeds * attn_mask.unsqueeze(-1).to(embeds.dtype)
        bank_cpu = embeds.detach().to("cpu", dtype=dtype).contiguous()
        del enc_runner, txt_enc, tok, input_ids, attn_mask, embeds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dist.is_initialized() and world_size > 1:
        if bank_cpu is None:
            bank_gpu = torch.empty((len(prompts), max_seq_len, text_dim), device=device, dtype=dtype)
        else:
            bank_gpu = bank_cpu.to(device)
        dist.broadcast(bank_gpu, src=0)
        bank_cpu = bank_gpu.cpu()
        del bank_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert bank_cpu is not None
    return bank_cpu


def _load_dit_model(
    model_name_or_path: str,
    *,
    subfolder: Optional[str],
    tiny: bool,
    dtype: torch.dtype,
) -> nn.Module:
    from diffusers import WanTransformer3DModel

    if tiny:
        cfg = WanTransformer3DModel.load_config(
            {
                "patch_size": [1, 2, 2],
                "num_attention_heads": 4,
                "attention_head_dim": 32,
                "in_channels": 16,
                "out_channels": 16,
                "text_dim": 64,
                "freq_dim": 64,
                "ffn_dim": 256,
                "num_layers": 2,
                "cross_attn_norm": True,
                "qk_norm": "rms_norm_across_heads",
                "eps": 1e-6,
            }
        )
        return WanTransformer3DModel.from_config(cfg).to(dtype)

    path_obj = Path(model_name_or_path)
    resolved_sub = subfolder
    if path_obj.is_dir() and resolved_sub and not (path_obj / resolved_sub).is_dir():
        resolved_sub = None

    cfg_file = (
        (path_obj / resolved_sub / "config.json")
        if (path_obj.is_dir() and resolved_sub)
        else (path_obj / "config.json")
    )
    cls_name = "WanTransformer3DModel"
    if cfg_file.is_file():
        meta = json.loads(cfg_file.read_text())
        cls_name = meta.get("_class_name", cls_name)

    import diffusers

    dit_cls = getattr(diffusers, cls_name, WanTransformer3DModel)
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if resolved_sub:
        load_kwargs["subfolder"] = resolved_sub
    return dit_cls.from_pretrained(model_name_or_path, **load_kwargs)


def _extract_config_and_cli(argv: list[str]) -> tuple[Optional[str], list[str]]:
    config_path: Optional[str] = None
    rem: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
            continue
        if a.startswith("--config="):
            config_path = a.split("=", 1)[1]
            i += 1
            continue
        rem.append(a)
        i += 1
    return config_path, rem


def _parse_cli_overrides(base: dict[str, Any], cli_argv: list[str], parser: HfArgumentParser) -> dict[str, Any]:
    merged = dict(base)
    action_map = {opt: act for act in parser._actions for opt in act.option_strings}
    i = 0
    while i < len(cli_argv):
        tok = cli_argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if "=" in tok:
            opt, val = tok.split("=", 1)
            vals = [val]
            step = 1
        else:
            opt = tok
            vals = []
            step = 1
            while i + step < len(cli_argv) and not cli_argv[i + step].startswith("--"):
                vals.append(cli_argv[i + step])
                step += 1
        act = action_map.get(opt)
        dest = act.dest if act is not None else opt.lstrip("-").replace("-", "_")
        if act is not None and act.nargs == 0:
            merged[dest] = bool(getattr(act, "const", True))
        elif not vals:
            merged[dest] = True
        elif act is not None and act.nargs in ("+", "*"):
            conv = act.type or str
            merged[dest] = [conv(v) for v in vals]
        else:
            raw_v = vals[0]
            if raw_v.lower() in ("true", "false") and (act is None or act.type in (bool, None)):
                merged[dest] = raw_v.lower() == "true"
            elif act is not None and act.type is not None:
                merged[dest] = act.type(raw_v)
            else:
                merged[dest] = raw_v
        i += step
    return merged


def _build_quant_config_from_yaml(quant_blk: Optional[dict[str, Any]]) -> FakeQuantConfig:
    if not quant_blk:
        return FakeQuantConfig.diffusion_mxfp4_fp8_hybrid()
    preset = quant_blk.get("preset", "diffusion_mxfp4_fp8_hybrid")
    backend = quant_blk.get("backend", "auto")
    if preset == "diffusion_mxfp4_fp8_hybrid":
        return FakeQuantConfig.diffusion_mxfp4_fp8_hybrid(backend=backend)
    return FakeQuantConfig.from_dict(quant_blk)


def run_diffusion_qad(argv: Optional[list[str]] = None) -> DiffusionQADTrainer:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config_path, cli_argv = _extract_config_and_cli(raw_argv)
    parser = HfArgumentParser((DiffusionModelDataArguments, DiffusionQADTrainingArguments))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        if world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")

    inline_quant: Optional[FakeQuantConfig] = None
    if config_path is not None:
        raw_yaml = yaml.safe_load(Path(config_path).read_text()) or {}
        inline_quant = _build_quant_config_from_yaml(raw_yaml.pop("quant", None))
        lora_blk = raw_yaml.pop("lora", None)
        if isinstance(lora_blk, dict):
            raw_yaml["use_lora"] = True
            raw_yaml.setdefault("lora_r", int(lora_blk.get("r", lora_blk.get("rank", 32))))
            raw_yaml.setdefault("lora_alpha", int(lora_blk.get("lora_alpha", lora_blk.get("alpha", 32))))
            raw_yaml.setdefault("lora_dropout", float(lora_blk.get("dropout", 0.0)))
            raw_yaml.setdefault(
                "lora_merge_before_quant", lora_blk.get("quant_order", "lora_then_quant") != "quant_then_lora"
            )
            if "target_modules" in lora_blk or "targets" in lora_blk:
                raw_yaml.setdefault("lora_target_modules", list(lora_blk.get("target_modules") or lora_blk["targets"]))

        raw_yaml.setdefault("output_dir", "runs/diffusion_qad_output")
        raw_yaml.setdefault("remove_unused_columns", False)
        raw_yaml.setdefault("report_to", [])
        merged = _parse_cli_overrides(raw_yaml, cli_argv, parser)
        if not torch.cuda.is_available():
            merged["bf16"] = False
            merged["use_cpu"] = True
            merged.pop("fsdp", None)
            merged.pop("fsdp_config", None)
        elif world <= 1:
            merged.pop("fsdp", None)
            merged.pop("fsdp_config", None)
        model_args, training_args = parser.parse_dict(merged, allow_extra_keys=True)
    else:
        model_args, training_args = parser.parse_args_into_dataclasses(cli_argv)
        inline_quant = training_args.build_quant_config() or FakeQuantConfig.diffusion_mxfp4_fp8_hybrid()

    dtype = torch.float32 if training_args.use_cpu else getattr(torch, model_args.model_dtype, torch.bfloat16)

    # Peek transformer config for in_channels & text_dim before encoding prompts so text_encoder is freed first
    in_channels, text_dim = (16, 64) if model_args.tiny else (48, 4096)
    if not model_args.tiny:
        p = Path(model_args.model_name_or_path)
        cfg_file = (
            (p / model_args.subfolder / "config.json") if (p.is_dir() and model_args.subfolder) else (p / "config.json")
        )
        if cfg_file.is_file():
            meta = json.loads(cfg_file.read_text())
            in_channels = int(meta.get("in_channels", in_channels))
            text_dim = int(meta.get("text_dim", meta.get("joint_attention_dim", text_dim)))

    if model_args.latents_dir:
        train_dataset: Dataset = DirectoryLatentDataset(model_args.latents_dir, dtype=dtype)
    else:
        prompt_bank = _encode_prompt_bank(
            model_args.model_name_or_path,
            prompts=DEFAULT_VIDEO_PROMPTS,
            max_seq_len=model_args.max_text_seq_len,
            text_dim=text_dim,
            dtype=dtype,
            enable_real_encoder=bool(model_args.encode_real_prompts and not model_args.tiny),
            local_rank=local_rank,
            world_size=world,
        )
        train_dataset = LatentPromptDataset(
            num_samples=model_args.num_samples,
            in_channels=in_channels,
            num_frames=model_args.latent_frames,
            height=model_args.latent_height,
            width=model_args.latent_width,
            prompt_embeds_bank=prompt_bank,
            dtype=dtype,
        )

    student = _load_dit_model(
        model_args.model_name_or_path,
        subfolder=model_args.subfolder,
        tiny=model_args.tiny,
        dtype=dtype,
    )

    if training_args.fsdp and isinstance(training_args.fsdp_config, dict):
        if not training_args.fsdp_config.get("transformer_layer_cls_to_wrap"):
            training_args.fsdp_config["transformer_layer_cls_to_wrap"] = decoder_layer_class_names(student)
        # Under FSDP2, wrap blocks with PyTorch `CheckpointWrapper` (outside FSDPModule) rather than
        # Diffusers's inner `_gradient_checkpointing_func` so MixedPrecisionPolicy casts match on recompute.
        if training_args.gradient_checkpointing:
            training_args.fsdp_config["activation_checkpointing"] = True
            training_args.gradient_checkpointing = False

    lora_cfg: Optional[dict[str, Any]] = None
    if model_args.use_lora:
        default_dit_targets = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "ffn.net.0.proj",
            "ffn.net.2",
            "ff.net.0.proj",
            "ff.net.2",
        ]
        all_names = [n for n, m in student.named_modules() if isinstance(m, nn.Linear)]
        candidates = list(model_args.lora_target_modules) if model_args.lora_target_modules else default_dit_targets
        targets = [t for t in candidates if any(n == t or n.endswith(f".{t}") for n in all_names)]
        if not targets:
            targets = list(candidates)
        lora_cfg = {
            "r": model_args.lora_r,
            "lora_alpha": model_args.lora_alpha,
            "lora_dropout": model_args.lora_dropout,
            "target_modules": targets,
            "bias": "none",
        }

    quant_config = inline_quant or FakeQuantConfig.diffusion_mxfp4_fp8_hybrid()
    student = prepare_diffusion_qad_model(
        student,
        quant_config,
        lora_config=lora_cfg,
        lora_merge_before_quant=training_args.lora_merge_before_quant,
    )

    if training_args.teacher_model_name_or_path == "self" or training_args.teacher_placement == "self":
        teacher_model: Union[nn.Module, str] = "self"
    else:
        t_path = training_args.teacher_model_name_or_path or model_args.model_name_or_path
        teacher_model = _load_dit_model(
            t_path,
            subfolder=model_args.subfolder,
            tiny=model_args.tiny,
            dtype=dtype,
        )

    diff_cfg = training_args.build_diffusion_qad_config()
    if model_args.feature_block_indices:
        diff_cfg.feature_block_indices = list(model_args.feature_block_indices)

    callbacks = []
    if model_args.stop_after_steps > 0:
        callbacks.append(StopAfterSteps(model_args.stop_after_steps))
    if model_args.grad_debug:
        callbacks.append(GradDebugCallback())

    trainer = DiffusionQADTrainer(
        model=student,
        args=training_args,
        teacher_model=teacher_model,
        diffusion_config=diff_cfg,
        quant_config=quant_config,
        train_dataset=train_dataset,
        data_collator=_collate_diffusion_batch,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(training_args.output_dir))
    return trainer


if __name__ == "__main__":
    run_diffusion_qad()
