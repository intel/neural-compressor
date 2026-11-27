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
"""Small CUDA/FSDP2 tests for v2.0 QAD (no model download or multi-GB allocation)."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

from safetensors import safe_open
import torch
from transformers import TrainingArguments

from neural_compressor.torch.distillation import DistillConfig
from neural_compressor.torch.qad import QADTrainer, prepare_qad_model
from neural_compressor.torch.quantization import (
    FakeQuantConfig,
    QuantSchemeRule,
    save_fsdp2_full_model,
    save_fsdp2_peft_adapter,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples/pytorch/llm/qwen/qad"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


def build_tiny_qwen35(**kwargs):
    path = EXAMPLE_DIR / "modeling/qwen35_text_loader.py"
    spec = importlib.util.spec_from_file_location("qad_example_qwen35_text_loader", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qad_example_qwen35_text_loader"] = module
    spec.loader.exec_module(module)
    return module.build_tiny_qwen35(**kwargs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGPUQAD(unittest.TestCase):
    def setUp(self):
        import torch.distributed as dist

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29651")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        self._created_pg = False
        if not dist.is_initialized():
            world_size = int(os.environ["WORLD_SIZE"])
            backend = "nccl" if world_size > 1 else "gloo"
            dist.init_process_group(backend, rank=int(os.environ["RANK"]), world_size=world_size)
            self._created_pg = world_size == 1

    def tearDown(self):
        import torch.distributed as dist

        if getattr(self, "_created_pg", False) and dist.is_initialized():
            dist.destroy_process_group()

    def _lora_model(self, quant_order):
        torch.manual_seed(0)
        model = build_tiny_qwen35(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            dtype=torch.bfloat16,
        )
        config = FakeQuantConfig(
            config_groups=[
                QuantSchemeRule(
                    targets=[r"re:.*self_attn\.q_proj$"],
                    weights="mxfp4",
                    input_activations="mxfp4",
                    output_activations="fp8_e4m3",
                )
            ]
        )
        model = prepare_qad_model(
            model,
            config,
            lora_config={"r": 4, "lora_alpha": 4, "target_modules": ["q_proj"], "bias": "none", "task_type": "CAUSAL_LM"},
            lora_merge_before_quant=(quant_order == "lora_then_quant"),
        )
        return model.to("cuda")

    def test_lora_a_and_b_backward(self):
        for order in ("quant_then_lora", "lora_then_quant"):
            model = self._lora_model(order)
            layer = model.base_model.model.model.layers[0].self_attn.q_proj
            output = layer(torch.randn(3, 64, device="cuda", dtype=torch.bfloat16))
            output.float().sum().backward()
            self.assertIsNotNone(layer.lora_A.default.weight.grad)
            self.assertIsNotNone(layer.lora_B.default.weight.grad)

    def test_fsdp2_save_adapter_and_full_state(self):
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        model = self._lora_model("lora_then_quant")
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )
        for layer in model.base_model.model.model.layers:
            fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
        fully_shard(model, mp_policy=policy, reshard_after_forward=False)
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp) / "adapter"
            full_dir = Path(tmp) / "full"
            self.assertTrue(save_fsdp2_peft_adapter(model, adapter_dir, should_save=True))
            self.assertTrue((adapter_dir / "adapter_model.safetensors").is_file())

            plain = build_tiny_qwen35(
                num_hidden_layers=2,
                layer_types=["full_attention", "full_attention"],
                dtype=torch.bfloat16,
            ).to("cuda")
            for layer in plain.model.layers:
                fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
            fully_shard(plain, mp_policy=policy, reshard_after_forward=False)
            self.assertTrue(save_fsdp2_full_model(plain, full_dir, should_save=True, dtype=torch.bfloat16))
            with safe_open(str(full_dir / "model.safetensors"), "pt") as handle:
                self.assertFalse(any("quantizer" in key for key in handle.keys()))

    def test_cpu_teacher_logits_work_with_cuda_student(self):
        student = self._lora_model("quant_then_lora")
        teacher = build_tiny_qwen35(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            dtype=torch.float32,
        ).cpu().eval()
        batch = {
            "input_ids": torch.randint(0, 128, (1, 16), device="cuda"),
            "attention_mask": torch.ones(1, 16, dtype=torch.long, device="cuda"),
            "loss_mask": torch.ones(1, 16, device="cuda"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            trainer = QADTrainer(
                model=student,
                teacher=teacher,
                args=TrainingArguments(output_dir=tmp, report_to=[], use_cpu=False, bf16=True),
                distill_config=DistillConfig(teacher_device="cpu", top_k=16, chunk=4),
            )
            loss = trainer.compute_loss(student, batch)
            self.assertEqual(loss.device.type, "cuda")
            loss.backward()
        self.assertIsNotNone(student.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.grad)

    def test_self_teacher_zero_vram_lora_step(self):
        student = self._lora_model("lora_then_quant")
        batch = {
            "input_ids": torch.randint(0, 128, (1, 16), device="cuda"),
            "attention_mask": torch.ones(1, 16, dtype=torch.long, device="cuda"),
            "loss_mask": torch.ones(1, 16, device="cuda"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            trainer = QADTrainer(
                model=student,
                teacher="self",
                args=TrainingArguments(output_dir=tmp, report_to=[], use_cpu=False, bf16=True),
                distill_config=DistillConfig(teacher="self", teacher_device="self", top_k=16, chunk=4),
            )
            self.assertIsNone(trainer.teacher)
            loss = trainer.compute_loss(student, batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
        self.assertIsNotNone(student.base_model.model.model.layers[0].self_attn.q_proj.lora_B.default.weight.grad)

    def test_trainer_fsdp2_resume(self):
        student = self._lora_model("lora_then_quant")
        teacher = build_tiny_qwen35(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            dtype=torch.bfloat16,
        ).to("cuda").eval()
        dataset = [
            {
                "input_ids": torch.randint(0, 128, (16,)),
                "attention_mask": torch.ones(16, dtype=torch.long),
                "loss_mask": torch.ones(16),
            }
            for _ in range(4)
        ]
        with tempfile.TemporaryDirectory() as local_tmp:
            import torch.distributed as dist

            obj_list = [local_tmp]
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.broadcast_object_list(obj_list, src=0)
            tmp = obj_list[0]

            def arguments(steps):
                return TrainingArguments(
                    output_dir=tmp,
                    per_device_train_batch_size=1,
                    max_steps=steps,
                    learning_rate=1e-4,
                    logging_steps=1,
                    save_steps=1,
                    save_strategy="steps",
                    save_total_limit=1,
                    report_to=[],
                    bf16=True,
                    use_cpu=False,
                    remove_unused_columns=False,
                    fsdp="full_shard auto_wrap",
                    fsdp_config={
                        "fsdp_version": 2,
                        "transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"],
                        "reshard_after_forward": True,
                    },
                )

            trainer = QADTrainer(
                model=student,
                teacher=teacher,
                args=arguments(1),
                train_dataset=dataset,
                distill_config=DistillConfig(top_k=16, chunk=4),
            )
            trainer.train()
            resumed_student = self._lora_model("lora_then_quant")
            resumed = QADTrainer(
                model=resumed_student,
                teacher=teacher,
                args=arguments(2),
                train_dataset=dataset,
                distill_config=DistillConfig(top_k=16, chunk=4),
            )
            resumed.train(resume_from_checkpoint=str(Path(tmp) / "checkpoint-1"))
            self.assertEqual(resumed.state.global_step, 2)
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.barrier()


if __name__ == "__main__":
    unittest.main()
