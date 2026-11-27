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
"""Qwen3.5 / Qwen3.8 QAD training entry point using ``HfArgumentParser`` + ``QADTrainer``.

Example invocations:
    # 8x GPU Full QAD (MXFP4 MLP + Static FP8 Attention/GDN)
    torchrun --nproc_per_node=8 train.py --config recipes/qwen3_5_27b_mxfp4_fp8_full_fsdp2.yaml \\
        --data_prefix /path/to/tokens_text_document --output_dir runs/qwen27b-full

    # 4x GPU Low-Memory Full QAD (< 82 GiB/GPU via FSDP2 CPU offload / 8-bit AdamW)
    torchrun --nproc_per_node=4 train.py --config recipes/qwen3_5_27b_mxfp4_fp8_full_cpu_offload.yaml \\
        --data_prefix /path/to/tokens_text_document --output_dir runs/qwen27b-lowmem

    # Zero-VRAM Self-Teacher LoRA Path-B QAD (< 45 GiB/GPU)
    torchrun --nproc_per_node=4 train.py --config recipes/qwen3_5_27b_mxfp4_fp8_lora_path_b.yaml \\
        --data_prefix /path/to/tokens_text_document --output_dir runs/qwen27b-lora
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from transformers import HfArgumentParser
import yaml

EXAMPLE_ROOT = Path(__file__).resolve().parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from data.megatron_indexed_dataset import PackedDataset, SyntheticDataset, default_collator  # noqa: E402
from modeling.qwen35_text_loader import load_qwen_causal_lm  # noqa: E402

from neural_compressor.torch.distillation import PrecomputedTopKCollator, PrecomputedTopKDataset  # noqa: E402
from neural_compressor.torch.export import export_hf_checkpoint  # noqa: E402
from neural_compressor.torch.qad import (  # noqa: E402
    GradDebugCallback,
    QADTrainer,
    QADTrainingArguments,
    StopAfterSteps,
    decoder_layer_class_names,
    prepare_qad_model,
)
from neural_compressor.torch.quantization.fake_quant import FakeQuantConfig  # noqa: E402


@dataclass
class ModelDataArguments:
    """Model, dataset, and optional LoRA CLI parameters for the Qwen QAD example."""

    model_name_or_path: str = field(
        default="Qwen/Qwen3.8-27B",
        metadata={"help": "HuggingFace CausalLM or VLM checkpoint directory or Hub ID."},
    )
    model_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model parameter dtype ('bfloat16', 'float16', or 'float32')."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow custom modeling code from remote Hub repositories."},
    )
    tiny: bool = field(
        default=False,
        metadata={"help": "Build a tiny 2-layer Qwen3.5 model for fast smoke testing."},
    )
    use_unsloth: bool = field(
        default=False,
        metadata={"help": "Load student via optional Unsloth FastLanguageModel backend."},
    )
    data_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Megatron .bin/.idx prefix (omit when using --synthetic_batches or --offline_topk_dir)."},
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Alias for --data_prefix."},
    )
    amax_source: Optional[str] = field(
        default=None,
        metadata={"help": "Optional quantizer_state JSON or ModelOpt .quant_summary.txt / .pt file."},
    )
    offline_topk_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory of precomputed offline teacher Top-K .pt shards."},
    )
    synthetic_batches: int = field(
        default=0,
        metadata={"help": "Number of deterministic synthetic sequences for smoke/memory benchmarking."},
    )
    keep_vision_on_export: bool = field(
        default=True,
        metadata={"help": "Preserve VLM vision tower weights when exporting a multimodal base checkpoint."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Attach PEFT LoRA adapters via prepare_qad_model."},
    )
    lora_r: int = field(default=64, metadata={"help": "LoRA rank r."})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA scaling alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout probability."})
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Target linear module suffixes for PEFT LoRA."},
    )


def _extract_config_path_and_remaining_argv(argv: list[str]) -> tuple[Optional[str], list[str]]:
    """Pop ``--config <path>`` (or ``--recipe <path>``) from ``argv`` while keeping CLI overrides."""
    config_path: Optional[str] = None
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--config", "--recipe") and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
        elif arg.startswith("--config=") or arg.startswith("--recipe="):
            config_path = arg.split("=", 1)[1]
            i += 1
        else:
            remaining.append(arg)
            i += 1
    return config_path, remaining


def _normalize_yaml_dict(raw: dict[str, Any], yaml_path: str) -> tuple[dict[str, Any], Optional[FakeQuantConfig]]:
    """Normalize either flat v2.0 YAML recipes or legacy nested ``model/distill/train`` YAML recipes."""
    flat: dict[str, Any] = {}
    inline_quant: Optional[FakeQuantConfig] = None

    if "quant" in raw and isinstance(raw["quant"], dict):
        inline_quant = FakeQuantConfig.from_dict(raw["quant"])
    elif "config_groups" in raw or "scheme" in raw:
        inline_quant = FakeQuantConfig.from_dict(raw)

    # Legacy nested recipe compatibility (model / distill / train / export blocks)
    if any(k in raw for k in ("model", "distill", "train")):
        model_blk = raw.get("model") or {}
        distill_blk = raw.get("distill") or {}
        train_blk = raw.get("train") or {}
        export_blk = raw.get("export") or {}

        if "path" in model_blk:
            flat["model_name_or_path"] = model_blk["path"]
        if "dtype" in model_blk:
            flat["model_dtype"] = model_blk["dtype"]
        if "tiny" in model_blk:
            flat["tiny"] = bool(model_blk["tiny"])
        if "trust_remote_code" in model_blk:
            flat["trust_remote_code"] = bool(model_blk["trust_remote_code"])

        if "teacher" in distill_blk:
            flat["teacher_model_name_or_path"] = distill_blk["teacher"]
        if "teacher_device" in distill_blk:
            flat["teacher_placement"] = distill_blk["teacher_device"]
        if "loss" in distill_blk:
            flat["kd_loss_type"] = distill_blk["loss"]
        if "top_k" in distill_blk:
            flat["kd_top_k"] = distill_blk["top_k"]
        if "temperature" in distill_blk:
            flat["kd_temperature"] = distill_blk["temperature"]
        if "reverse" in distill_blk:
            flat["kd_reverse_kl"] = distill_blk["reverse"]
        if "kd_weight" in distill_blk:
            flat["kd_loss_weight"] = distill_blk["kd_weight"]
        if "ce_weight" in distill_blk:
            flat["ce_loss_weight"] = distill_blk["ce_weight"]
        if "chunk" in distill_blk or "chunk_size" in distill_blk:
            flat["logits_chunk_size"] = distill_blk.get("chunk_size", distill_blk.get("chunk", 1024))

        world = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        mbs = int(train_blk.get("micro_batch_size", 1))
        gbs = int(train_blk.get("global_batch_size", mbs * world))
        flat["per_device_train_batch_size"] = mbs
        flat["gradient_accumulation_steps"] = max(1, gbs // max(1, mbs * world))
        if "seq_length" in train_blk:
            flat["seq_length"] = train_blk["seq_length"]
        if "train_iters" in train_blk:
            flat["max_steps"] = train_blk["train_iters"]
        if "lr" in train_blk:
            flat["learning_rate"] = float(train_blk["lr"])
        if "min_lr" in train_blk:
            flat["lr_scheduler_type"] = "cosine_with_min_lr"
            flat["lr_scheduler_kwargs"] = {"min_lr": float(train_blk["min_lr"])}
        if "warmup" in train_blk:
            flat["warmup_steps"] = int(train_blk["warmup"])
        if "weight_decay" in train_blk:
            flat["weight_decay"] = float(train_blk["weight_decay"])
        if "clip_grad" in train_blk:
            flat["max_grad_norm"] = float(train_blk["clip_grad"])
        if "adam_beta2" in train_blk:
            flat["adam_beta2"] = float(train_blk["adam_beta2"])
        if "adam_eps" in train_blk:
            flat["adam_epsilon"] = float(train_blk["adam_eps"])
        if "seed" in train_blk:
            flat["seed"] = int(train_blk["seed"])
        if "logging_steps" in train_blk:
            flat["logging_steps"] = int(train_blk["logging_steps"])
        if "save_steps" in train_blk:
            save_steps = int(train_blk["save_steps"])
            flat["save_strategy"] = "steps" if save_steps > 0 else "no"
            flat["save_steps"] = max(save_steps, 1)
        if train_blk.get("backend") == "unsloth":
            flat["use_unsloth"] = True
        gc_enabled = bool(train_blk.get("gradient_checkpointing", False))
        gc_mode = str(train_blk.get("gc_mode", "trainer"))
        if train_blk.get("parallel") == "fsdp2" and world > 1:
            flat["fsdp"] = "full_shard auto_wrap"
            flat["fsdp_config"] = {
                "fsdp_version": 2,
                "reshard_after_forward": True,
                "activation_checkpointing": gc_enabled and gc_mode == "fsdp",
            }
            flat["gradient_checkpointing"] = gc_enabled and gc_mode == "trainer"
        else:
            flat["gradient_checkpointing"] = gc_enabled

        lora_blk = train_blk.get("lora")
        if isinstance(lora_blk, dict):
            flat["use_lora"] = True
            flat["lora_r"] = int(lora_blk.get("rank", lora_blk.get("r", 64)))
            flat["lora_alpha"] = int(lora_blk.get("alpha", lora_blk.get("lora_alpha", 64)))
            flat["lora_dropout"] = float(lora_blk.get("dropout", 0.0))
            flat["lora_merge_before_quant"] = lora_blk.get("quant_order", "lora_then_quant") != "quant_then_lora"
            if "targets" in lora_blk:
                flat["lora_target_modules"] = list(lora_blk["targets"])

        if "format" in export_blk:
            fmt = str(export_blk["format"])
            flat["export_format"] = "bf16_qdq" if fmt in ("hf_bf16", "bf16", "bf16_qdq") else "compressed_tensors"
        if "keep_vision" in export_blk:
            flat["keep_vision_on_export"] = bool(export_blk["keep_vision"])

    # Overlay any top-level flat v2.0 keys directly
    reserved = {"schema", "name", "quant", "model", "distill", "train", "export", "lora"}
    for key, val in raw.items():
        if key not in reserved:
            flat[key] = val

    if isinstance(raw.get("lora"), dict):
        lora_top = raw["lora"]
        flat["use_lora"] = True
        flat.setdefault("lora_r", int(lora_top.get("r", lora_top.get("rank", 64))))
        flat.setdefault("lora_alpha", int(lora_top.get("lora_alpha", lora_top.get("alpha", 64))))
        flat.setdefault("lora_dropout", float(lora_top.get("lora_dropout", lora_top.get("dropout", 0.0))))
        if "target_modules" in lora_top or "targets" in lora_top:
            flat.setdefault("lora_target_modules", list(lora_top.get("target_modules") or lora_top["targets"]))

    flat.setdefault("output_dir", "runs/qad_output")
    flat.setdefault("remove_unused_columns", False)
    flat.setdefault("report_to", [])
    if inline_quant is not None and inline_quant.needs_calibration() and not flat.get("quantizer_state_path"):
        flat.setdefault("calib_batches", 8)
    return flat, inline_quant


def _parse_cli_overrides_into_dict(base: dict[str, Any], cli_argv: list[str], parser: HfArgumentParser) -> dict[str, Any]:
    """Apply explicit CLI flags on top of ``base`` dictionary before ``parser.parse_dict``."""
    merged = dict(base)
    action_map: dict[str, Any] = {}
    for action in parser._actions:
        for opt in action.option_strings:
            action_map[opt] = action

    i = 0
    while i < len(cli_argv):
        token = cli_argv[i]
        if not token.startswith("--"):
            i += 1
            continue
        if "=" in token:
            opt, val_str = token.split("=", 1)
            next_tokens = [val_str]
            step = 1
        else:
            opt = token
            step = 1
            next_tokens = []
            while i + step < len(cli_argv) and not cli_argv[i + step].startswith("--"):
                next_tokens.append(cli_argv[i + step])
                step += 1

        action = action_map.get(opt)
        dest = action.dest if action is not None else opt.lstrip("-").replace("-", "_")
        if action is not None and action.nargs == 0:
            merged[dest] = bool(getattr(action, "const", True))
        elif len(next_tokens) == 0:
            merged[dest] = True
        elif action is not None and action.nargs in ("+", "*"):
            conv = action.type or str
            merged[dest] = [conv(v) for v in next_tokens]
        else:
            raw_v = next_tokens[0]
            if raw_v.lower() in ("true", "false") and (action is None or action.type in (bool, None)):
                merged[dest] = raw_v.lower() == "true"
            elif action is not None and action.type is not None:
                merged[dest] = action.type(raw_v)
            else:
                merged[dest] = raw_v
        i += step
    return merged


def parse_args(argv: Optional[list[str]] = None) -> tuple[ModelDataArguments, QADTrainingArguments, Optional[FakeQuantConfig]]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config_path, cli_argv = _extract_config_path_and_remaining_argv(raw_argv)
    parser = HfArgumentParser((ModelDataArguments, QADTrainingArguments))

    inline_quant: Optional[FakeQuantConfig] = None
    if config_path is not None:
        raw_yaml = yaml.safe_load(Path(config_path).read_text()) or {}
        base_dict, inline_quant = _normalize_yaml_dict(raw_yaml, config_path)
        merged_dict = _parse_cli_overrides_into_dict(base_dict, cli_argv, parser)
        if not torch.cuda.is_available() and not (getattr(torch, "xpu", None) and torch.xpu.is_available()):
            merged_dict["bf16"] = False
            merged_dict["use_cpu"] = True
            merged_dict.pop("fsdp", None)
            merged_dict.pop("fsdp_config", None)
        elif int(os.environ.get("WORLD_SIZE", "1")) <= 1:
            merged_dict.pop("fsdp", None)
            merged_dict.pop("fsdp_config", None)
        model_args, training_args = parser.parse_dict(merged_dict, allow_extra_keys=True)
    else:
        model_args, training_args = parser.parse_args_into_dataclasses(cli_argv)

    return model_args, training_args, inline_quant


def _resolve_dtype(dtype_name: str, use_cpu: bool) -> torch.dtype:
    if use_cpu:
        return torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[dtype_name.lower()]


def run_training(argv: Optional[list[str]] = None) -> QADTrainer:
    model_args, training_args, inline_quant = parse_args(argv)
    if model_args.data_path and not model_args.data_prefix:
        model_args.data_prefix = model_args.data_path
    amax_source = model_args.amax_source or training_args.quantizer_state_path
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        if world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
    if model_args.use_unsloth and world > 1:
        os.environ.setdefault("UNSLOTH_FSDP2", "1")

    dtype = _resolve_dtype(model_args.model_dtype, bool(training_args.use_cpu))
    student = load_qwen_causal_lm(
        model_args.model_name_or_path,
        dtype=dtype,
        tiny=model_args.tiny,
        trust_remote_code=model_args.trust_remote_code,
        use_unsloth=model_args.use_unsloth,
        max_seq_length=training_args.seq_length,
    )

    # Auto-populate FSDP2 decoder layer class names from the loaded student before wrapping
    if training_args.fsdp and isinstance(training_args.fsdp_config, dict):
        if not training_args.fsdp_config.get("transformer_layer_cls_to_wrap"):
            training_args.fsdp_config["transformer_layer_cls_to_wrap"] = decoder_layer_class_names(student)

    quant_config = inline_quant or training_args.build_quant_config()
    if quant_config is not None:
        if training_args.quant_backend and training_args.quant_backend != "auto":
            quant_config.backend = training_args.quant_backend  # type: ignore[assignment]
        if amax_source:
            quant_config.quantizer_state_path = None

    lora_cfg: Optional[dict[str, Any]] = None
    if model_args.use_lora:
        targets = model_args.lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
        ]
        lora_cfg = {
            "r": model_args.lora_r,
            "lora_alpha": model_args.lora_alpha,
            "lora_dropout": model_args.lora_dropout,
            "target_modules": targets,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    if quant_config is not None:
        student = prepare_qad_model(
            student,
            quant_config,
            lora_config=lora_cfg,
            lora_merge_before_quant=training_args.lora_merge_before_quant,
        )
        if amax_source:
            import importlib.util
            from neural_compressor.torch.quantization.fake_quant.calibrate import load_quantizer_state_dict

            conv_spec = importlib.util.spec_from_file_location(
                "nc_qad_convert_modelopt_amax", EXAMPLE_ROOT / "tools" / "convert_modelopt_amax.py"
            )
            assert conv_spec is not None and conv_spec.loader is not None
            conv_mod = importlib.util.module_from_spec(conv_spec)
            conv_spec.loader.exec_module(conv_mod)
            payload = conv_mod.convert_modelopt_amax(amax_source)
            load_quantizer_state_dict(student, payload, strict=False)
            training_args.calib_batches = 0
            training_args.quantizer_state_path = None

    # Resolve teacher model ('self', offline shards -> None, or online CausalLM)
    distill_cfg = training_args.build_distill_config()
    if model_args.offline_topk_dir:
        distill_cfg.teacher_device = "offline"
        distill_cfg.offline_topk_dir = model_args.offline_topk_dir
        teacher_model: Optional[Any] = None
    elif distill_cfg.teacher == "self" or distill_cfg.teacher_device == "self":
        teacher_model = "self"
    else:
        teacher_path = str(distill_cfg.teacher or model_args.model_name_or_path)
        teacher_model = load_qwen_causal_lm(
            teacher_path,
            dtype=dtype,
            tiny=model_args.tiny,
            trust_remote_code=model_args.trust_remote_code,
            use_unsloth=False,
            max_seq_length=training_args.seq_length,
        )

    # Build dataset and collator
    if model_args.offline_topk_dir:
        train_dataset = PrecomputedTopKDataset(model_args.offline_topk_dir)
        data_collator = PrecomputedTopKCollator()
    elif model_args.synthetic_batches > 0 or model_args.tiny and not model_args.data_prefix:
        vocab_sz = int(getattr(student.config, "vocab_size", 128))
        n_samples = max(
            model_args.synthetic_batches,
            max(1, training_args.max_steps)
            * training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world,
        )
        train_dataset = SyntheticDataset(n_samples, training_args.seq_length, vocab_size=min(vocab_sz, 4096))
        data_collator = default_collator
    else:
        if not model_args.data_prefix:
            raise ValueError("Provide `--data_prefix <prefix>`, `--offline_topk_dir <dir>`, or `--synthetic_batches N`.")
        train_dataset = PackedDataset(model_args.data_prefix, training_args.seq_length)
        data_collator = default_collator

    callbacks = []
    if training_args.stop_after_steps > 0:
        callbacks.append(StopAfterSteps(training_args.stop_after_steps))
    if training_args.grad_debug:
        callbacks.append(GradDebugCallback())

    trainer = QADTrainer(
        model=student,
        args=training_args,
        teacher_model=teacher_model,
        distill_config=distill_cfg,
        quant_config=quant_config,
        calib_batches=training_args.calib_batches,
        quantizer_state_path=training_args.quantizer_state_path,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    save_strat = str(getattr(training_args.save_strategy, "value", training_args.save_strategy)).lower()
    if save_strat != "no" or training_args.export_dir:
        trainer.save_model(training_args.output_dir)

    if training_args.export_dir and trainer.args.should_save and not model_args.tiny:
        export_hf_checkpoint(
            training_args.output_dir,
            training_args.export_dir,
            export_format="bf16" if training_args.export_format in ("bf16", "bf16_qdq") else "compressed_tensors",
            quant_config=quant_config,
            base_model_path=model_args.model_name_or_path,
            keep_vision=model_args.keep_vision_on_export,
        )

    return trainer


if __name__ == "__main__":
    run_training()
