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
"""Unit tests for distillation and QAD (CPU, full-attention tiny model)."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_compressor.torch.distill import DistillConfig, TopKLogitsKLLoss, topk_kl
from neural_compressor.torch.qad import (
    QADConfig,
    QADTrainer,
    build_tiny_qwen35,
    export,
    get_adapter,
    write_quantization_config,
)
from neural_compressor.torch.qad.config import ExportConfig
from neural_compressor.torch.quantization import FakeQuantConfig, QuantRule, quantize


def tiny_model(dtype=torch.float32):
    return build_tiny_qwen35(
        num_hidden_layers=2,
        layer_types=["full_attention", "full_attention"],
        dtype=dtype,
    )


def tiny_quant_config():
    return FakeQuantConfig(
        rules=[
            QuantRule(module=r".*\.mlp\.(gate_proj|up_proj|down_proj)$", weight="mxfp4", input="mxfp4"),
            QuantRule(module=r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$", weight="fp8_e4m3"),
        ]
    )


class TestDistillLoss(unittest.TestCase):
    def test_topk_equals_full_kl_when_k_is_vocab(self):
        torch.manual_seed(0)
        student = torch.randn(2, 8, 16)
        teacher = torch.randn(2, 8, 16)
        loss = topk_kl(student, teacher, k=16)
        reference = F.kl_div(
            F.log_softmax(student, dim=-1),
            F.log_softmax(teacher, dim=-1),
            log_target=True,
            reduction="none",
        ).sum(dim=-1)
        torch.testing.assert_close(loss, reference)

    def test_chunked_loss_matches_full(self):
        torch.manual_seed(1)
        student = torch.randn(2, 16, 32)
        teacher = torch.randn(2, 16, 32)
        mask = torch.ones(2, 16)
        full = TopKLogitsKLLoss(top_k=8, chunk=0)
        chunked = TopKLogitsKLLoss(top_k=8, chunk=5)
        total_full, count_full = full(student, teacher, mask)
        total_chunk, count_chunk = chunked(student, teacher, mask)
        torch.testing.assert_close(total_full, total_chunk)
        torch.testing.assert_close(count_full, count_chunk)


class TestAdapters(unittest.TestCase):
    def test_generic_adapter_on_causal_lm(self):
        model = tiny_model()
        adapter = get_adapter(model)
        self.assertEqual(len(adapter.layers(model)), 2)
        self.assertIs(adapter.text_model(model), model.model)
        self.assertIs(adapter.lm_head(model), model.lm_head)
        self.assertEqual(adapter.wrap_classes(model), ["Qwen3_5DecoderLayer"])

    def test_registry_family(self):
        model = tiny_model()
        self.assertEqual(get_adapter("qwen3_5").family, "qwen3_5")
        self.assertEqual(get_adapter("unknown_family").family, "generic")
        self.assertEqual(get_adapter(model).family, "qwen3_5")


class TestQADConfig(unittest.TestCase):
    def test_yaml_roundtrip(self):
        recipe = {
            "schema": "neural_compressor.qad/v1",
            "model": {"path": "/tmp/model", "family": "qwen3_5"},
            "quant": {
                "rules": [
                    {
                        "module": r".*\.mlp\.(gate_proj|up_proj|down_proj)$",
                        "weight": {"format": "mxfp4"},
                        "input": {"format": "mxfp4"},
                    }
                ],
                "algorithm": "max",
                "exclude": ["^lm_head$"],
                "amax_source": None,
                "strict": True,
            },
            "distill": {"teacher": "/tmp/model", "top_k": 64, "chunk": 128},
            "train": {"seq_length": 128, "micro_batch_size": 1, "global_batch_size": 2},
            "export": {"format": "hf_bf16", "keep_vision": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipe.yaml"
            import yaml

            path.write_text(yaml.safe_dump(recipe))
            config = QADConfig.from_yaml(path)
        self.assertEqual(config.model.family, "qwen3_5")
        self.assertEqual(config.quant.rules[0].weight.format.name, "mxfp4")
        self.assertEqual(config.distill.top_k, 64)
        self.assertEqual(config.train.grad_accum, 2)
        self.assertTrue(config.export.keep_vision)
        self.assertEqual(config.to_dict()["schema"], "neural_compressor.qad/v1")


class TestQADTrainer(unittest.TestCase):
    def _trainer(self, student, teacher):
        from transformers import TrainingArguments

        args = TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            report_to=[],
            per_device_train_batch_size=1,
            use_cpu=True,
            bf16=False,
        )
        return QADTrainer(
            model=student,
            teacher=teacher,
            args=args,
            distill_config=DistillConfig(teacher="tiny", top_k=16, chunk=4),
        )

    def test_compute_loss_matches_manual_formula(self):
        torch.manual_seed(0)
        teacher = tiny_model()
        student = quantize(tiny_model(), tiny_quant_config())
        trainer = self._trainer(student, teacher)

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "loss_mask": torch.ones(2, 16),
        }
        loss = trainer.compute_loss(student, batch)

        with torch.no_grad():
            student_hidden = student(batch["input_ids"], batch["attention_mask"])
            teacher_hidden = teacher(batch["input_ids"], batch["attention_mask"])
            student_logits = student.logits(student_hidden)
            teacher_logits = teacher.logits(teacher_hidden)
            manual = topk_kl(student_logits, teacher_logits, k=16).mean()
        torch.testing.assert_close(loss, manual)

    def test_compute_loss_is_finite_and_backward_runs(self):
        torch.manual_seed(1)
        teacher = tiny_model()
        student = quantize(tiny_model(), tiny_quant_config())
        trainer = self._trainer(student, teacher)
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "loss_mask": torch.ones(2, 16),
        }
        loss = trainer.compute_loss(student, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(student.model.layers[0].mlp.gate_proj.weight.grad)


class TestQuantMetadataAndExport(unittest.TestCase):
    def test_write_quantization_config(self):
        model = quantize(tiny_model(), tiny_quant_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = write_quantization_config(model, tmp, tiny_quant_config())
            payload = json.loads(Path(path).read_text())
        self.assertEqual(payload["schema"], "neural_compressor.qad/v1")
        self.assertEqual(payload["mode"], "fake_quant")
        self.assertIn("rules", payload["quant"])

    def test_export_text(self):
        model = quantize(tiny_model(), tiny_quant_config())
        with tempfile.TemporaryDirectory() as tmp:
            out = export(model, None, ExportConfig(keep_vision=False), tmp)
            files = {item.name for item in Path(out).iterdir()}
            self.assertIn("model.safetensors", files)
            self.assertIn("config.json", files)


if __name__ == "__main__":
    unittest.main()
