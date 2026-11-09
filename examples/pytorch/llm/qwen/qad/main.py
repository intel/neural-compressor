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
"""Qwen3.8-27B QAD reference run: torchrun + HF Trainer + neural_compressor.torch.qad.

The CLI is intentionally thin: it only builds configs, loads the models and
hands everything to the library. Recipe files live under ``recipes/``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from neural_compressor.torch.qad import (
    GradDebugCallback,
    QADConfig,
    QADTrainer,
    StopAfterSteps,
    apply_lora,
    decoder_layer_class_names,
    default_collator,
    export,
    make_loader,
)
from neural_compressor.torch.qad.backends import load_backend_model
from neural_compressor.torch.qad.data import PackedDataset
from neural_compressor.torch.quantization import quantize


class SyntheticDataset(torch.utils.data.Dataset):
    """Deterministic random tokens for wiring smokes (vocab 128, tiny models)."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int = 128) -> None:
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(index)
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,), generator=generator)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "loss_mask": torch.ones_like(input_ids, dtype=torch.float32),
        }


def parse() -> argparse.Namespace:
    """Parse the command line; recipe values can be overridden here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="QAD recipe YAML")
    parser.add_argument("--data", default="", help="tokenized .bin/.idx prefix (not needed with --synthetic-batches)")
    parser.add_argument("--output", required=True, help="checkpoint directory")
    parser.add_argument("--model", default=None, help="override recipe model.path")
    parser.add_argument("--teacher", default=None, help="override recipe distill.teacher")
    parser.add_argument("--stop-after", type=int, default=0, help="stop after N steps, keep the LR schedule")
    parser.add_argument("--save-steps", type=int, default=None, help="override train.save_steps")
    parser.add_argument("--seq-length", type=int, default=None, help="override train.seq_length")
    parser.add_argument("--train-iters", type=int, default=None, help="override train.train_iters")
    parser.add_argument("--calib-batches", type=int, default=8, help="batches for amax calibration when needed")
    parser.add_argument("--synthetic-batches", type=int, default=0, help="use random tokens instead of --data (smoke)")
    parser.add_argument("--top-k", type=int, default=None, help="override distill.top_k")
    parser.add_argument("--tiny", action="store_true", help="tiny random model for a smoke run")
    parser.add_argument("--debug-grads", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--export-dir", default=None, help="export a HF directory after training (non-FSDP only)")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> QADConfig:
    """Load the recipe and apply CLI overrides."""
    config = QADConfig.from_yaml(args.recipe)
    if args.model:
        config.model.path = args.model
        if config.distill.teacher in (None, ""):
            config.distill.teacher = args.model
    if args.teacher:
        config.distill.teacher = args.teacher
    if args.save_steps is not None:
        config.train.save_steps = args.save_steps
    if args.seq_length is not None:
        config.train.seq_length = args.seq_length
    if args.train_iters is not None:
        config.train.train_iters = args.train_iters
    if args.top_k is not None:
        config.distill.top_k = args.top_k
    config.train.output_dir = args.output
    if args.tiny:
        config.model.tiny = True
        config.distill.teacher = args.model or config.model.path
        config.train.parallel = "none"
        config.train.gradient_checkpointing = False
    return config


def setup_distributed(config: QADConfig) -> tuple[int, int, torch.device]:
    """Set the local device and initialize NCCL when the teacher will be sharded."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if config.train.backend == "unsloth" and config.train.parallel == "fsdp2":
        os.environ.setdefault("UNSLOTH_FSDP2", "1")
    if config.train.parallel == "fsdp2" and world > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group("nccl")
    return local_rank, world, device


def _model_dtype(config: QADConfig, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return dtypes[config.model.dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported model.dtype {config.model.dtype!r}") from exc


def check_batch(config: QADConfig, world: int) -> int:
    """Validate gbs and return the gradient-accumulation factor."""
    batch = config.train.micro_batch_size * world
    if config.train.global_batch_size % batch:
        raise ValueError(f"global_batch_size={config.train.global_batch_size} not divisible by mbs*world={batch}")
    return config.train.global_batch_size // batch


def quantize_student(config: QADConfig, args: argparse.Namespace, local_rank: int, world: int, device):
    """Load and fake-quantize the student; calibrate only when no amax exists."""
    student = load_backend_model(
        config.train.backend,
        config.model.path,
        dtype=_model_dtype(config, device),
        tiny=config.model.tiny,
        trust_remote_code=config.model.trust_remote_code,
        max_seq_length=config.train.seq_length,
    )
    if not config.quant.needs_calibration():
        return quantize(student, config.quant)

    if world > 1:
        raise ValueError(
            "static FP8 quantizers need an amax file for multi-rank runs; "
            "set quant.amax_source or calibrate on a single rank first"
        )
    loader, _ = make_loader(
        args.data,
        seq_length=config.train.seq_length,
        micro_batch_size=1,
        world_size=1,
        rank=local_rank,
        shuffle=False,
    )
    student.to(device)

    def run_fn(model, data_loader, max_batches):
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(data_loader):
                model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                if index + 1 >= max_batches:
                    break

    return quantize(student, config.quant, run_fn=run_fn, run_args=(loader, args.calib_batches))


def load_teacher(config: QADConfig, world: int, device: torch.device):
    """Load the teacher and honor shard/cuda/cpu placement before its forward."""
    if isinstance(config.distill.teacher, torch.nn.Module):
        teacher = config.distill.teacher
    else:
        teacher_path = config.distill.teacher or config.model.path
        teacher = load_backend_model(
            config.train.backend,
            teacher_path,
            dtype=_model_dtype(config, device),
            tiny=config.model.tiny,
            trust_remote_code=config.model.trust_remote_code,
            max_seq_length=config.train.seq_length,
        )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    if config.distill.teacher_device == "shard" and config.train.parallel == "fsdp2" and world > 1:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )
        # Mirror the reference TP layout: shard layers, keep the small root
        # (embed/lm_head) replicated so the first forward does not unshard all.
        for layer in teacher.layers:
            fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=True)
    if config.distill.teacher_device == "cpu":
        teacher.to(torch.device("cpu"))
    else:
        teacher.to(device)
    return teacher


def build_arguments(config: QADConfig, grad_accum: int, student, device: torch.device):
    """Translate ``TrainConfig`` into HF ``TrainingArguments``."""
    from transformers import TrainingArguments

    fsdp_kwargs = {}
    if config.train.parallel == "fsdp2":
        fsdp_kwargs = {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "fsdp_version": 2,
                "transformer_layer_cls_to_wrap": decoder_layer_class_names(student),
                "reshard_after_forward": True,
                "activation_checkpointing": config.train.gradient_checkpointing and config.train.gc_mode == "fsdp",
            },
        }
    use_bf16 = device.type == "cuda"
    train = config.train
    return TrainingArguments(
        output_dir=train.output_dir,
        per_device_train_batch_size=train.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=train.train_iters,
        learning_rate=train.lr,
        max_grad_norm=train.clip_grad,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": train.min_lr},
        warmup_steps=train.warmup,
        seed=train.seed,
        weight_decay=train.weight_decay,
        adam_beta1=0.9,
        adam_beta2=train.adam_beta2,
        adam_epsilon=train.adam_eps,
        logging_steps=train.logging_steps,
        save_strategy="steps" if train.save_steps > 0 else "no",
        save_steps=max(train.save_steps, 1),
        save_total_limit=2,
        report_to=[],
        bf16=use_bf16,
        gradient_checkpointing=(
            train.gradient_checkpointing and (train.parallel != "fsdp2" or train.gc_mode == "trainer")
        ),
        use_cpu=not use_bf16,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        **fsdp_kwargs,
    )


def main() -> None:
    """Run training and optionally export."""
    args = parse()
    config = build_config(args)
    local_rank, world, device = setup_distributed(config)
    grad_accum = check_batch(config, world)

    student = quantize_student(config, args, local_rank, world, device)
    if config.train.lora:
        student = apply_lora(student, config.train.lora, backend=config.train.backend)

    teacher = load_teacher(config, world, device)
    if args.synthetic_batches:
        dataset = SyntheticDataset(args.synthetic_batches, config.train.seq_length)
    else:
        if not args.data:
            raise ValueError("pass --data (bin/idx prefix) or --synthetic-batches for a smoke run")
        dataset = PackedDataset(args.data, config.train.seq_length)

    callbacks = []
    if args.debug_grads:
        callbacks.append(GradDebugCallback())
    if args.stop_after > 0:
        callbacks.append(StopAfterSteps(args.stop_after))

    trainer = QADTrainer(
        model=student,
        teacher=teacher,
        args=build_arguments(config, grad_accum, student, device),
        train_dataset=dataset,
        data_collator=default_collator,
        qad_config=config,
        quant_config=config.quant,
        mode="lora" if config.train.lora else "full",
        backend=config.train.backend,
        callbacks=callbacks or None,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.export_dir:
        if config.train.parallel == "fsdp2":
            raise ValueError(
                "export from a saved checkpoint in a separate process when using FSDP2: "
                "load the checkpoint with the same recipe, then call qad.export()"
            )
        export(student, None, config.export, args.export_dir, base_model=config.model.path)

    trainer.save_state()
    Path(config.train.output_dir, "run_args.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in vars(args).items()) + "\n", encoding="utf-8"
    )
    if trainer.is_world_process_zero():
        print(json.dumps({"output": config.train.output_dir, "world_size": world}, indent=2), flush=True)


if __name__ == "__main__":
    main()
