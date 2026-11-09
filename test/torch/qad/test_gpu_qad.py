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
"""Small CUDA/FSDP2 tests; no model download or multi-GB allocation."""

import tempfile
import unittest
from pathlib import Path

import torch

from neural_compressor.torch.qad import apply_lora, build_tiny_qwen35
from neural_compressor.torch.qad.checkpoint import save_adapter, save_full_state_dict
from neural_compressor.torch.qad.export import load_state_dict_into_model
from neural_compressor.torch.quantization import FakeQuantConfig, QuantRule, quantize


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGPUQAD(unittest.TestCase):
    def setUp(self):
        import os
        import torch.distributed as dist

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29651")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not dist.is_initialized():
            dist.init_process_group("gloo", rank=0, world_size=1)

    def tearDown(self):
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    def _lora_model(self, quant_order):
        torch.manual_seed(0)
        model = build_tiny_qwen35(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            dtype=torch.bfloat16,
        )
        config = FakeQuantConfig(
            rules=[QuantRule(module=r".*self_attn.q_proj$", weight="mxfp4", input="mxfp4", output="fp8_e4m3")]
        )
        model = quantize(model, config)
        model = apply_lora(
            model,
            {"rank": 4, "alpha": 4, "targets": ["q_proj"], "quant_order": quant_order},
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
            self.assertTrue(save_adapter(model, tmp, should_save=True))
            self.assertTrue((Path(tmp) / "adapter_model.safetensors").is_file())
            save_full_state_dict(model, tmp, should_save=True)
            reloaded = build_tiny_qwen35(
                num_hidden_layers=2,
                layer_types=["full_attention", "full_attention"],
                dtype=torch.bfloat16,
            )
            load_state_dict_into_model(reloaded, Path(tmp) / "model.safetensors")
            self.assertFalse(any("quantizer" in key for key in reloaded.state_dict()))
    def test_cpu_teacher_logits_work_with_cuda_student(self):
        from neural_compressor.torch.distill import DistillConfig
        from neural_compressor.torch.qad import QADTrainer
        from transformers import TrainingArguments

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

    def test_trainer_fsdp2_resume(self):
        from neural_compressor.torch.distill import DistillConfig
        from neural_compressor.torch.qad import QADTrainer
        from transformers import TrainingArguments

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
        with tempfile.TemporaryDirectory() as tmp:
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


if __name__ == "__main__":
    unittest.main()
