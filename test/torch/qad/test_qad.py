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
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neural_compressor.torch.distill import DistillConfig, TopKLogitsKLLoss, topk_kl
from neural_compressor.torch.qad import (
    QADConfig,
    QADTrainer,
    apply_lora,
    build_tiny_qwen35,
    export,
    get_adapter,
    write_quantization_config,
)
from neural_compressor.torch.qad.config import ExportConfig
from neural_compressor.torch.qad.export import load_state_dict_into_model
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


def load_qad_example(filename):
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "examples/pytorch/llm/qwen/qad" / filename
    spec = importlib.util.spec_from_file_location(f"qad_example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPackedDataset(unittest.TestCase):
    def test_tail_mask_excludes_padding(self):
        from neural_compressor.torch.qad.data import PackedDataset

        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "tokens"
            tokens = np.arange(10, dtype=np.int32)
            lengths = np.array([4, 6], dtype=np.int32)
            with open(f"{prefix}.bin", "wb") as handle:
                handle.write(tokens.tobytes())
            with open(f"{prefix}.idx", "wb") as handle:
                handle.write(b"MMIDIDX\x00\x00")
                handle.write(struct.pack("<Q", 1))
                handle.write(struct.pack("<B", 4))
                handle.write(struct.pack("<Q", 2))
                handle.write(struct.pack("<Q", 2))
                handle.write(lengths.tobytes())
                handle.write(np.array([0, 4], dtype=np.int64).tobytes())
                handle.write(np.array([0, 4], dtype=np.int64).tobytes())
            sample = PackedDataset(str(prefix), 4, drop_tail=False)[2]
        self.assertEqual(sample["input_ids"].tolist(), [8, 9, 0, 0])
        self.assertEqual(sample["attention_mask"].tolist(), [1, 1, 0, 0])
        self.assertEqual(sample["loss_mask"].tolist(), [1.0, 1.0, 0.0, 0.0])


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

    def test_invalid_runtime_choices_are_rejected(self):
        from neural_compressor.torch.qad.config import TrainConfig

        with self.assertRaisesRegex(ValueError, "unknown backend"):
            TrainConfig(backend="modelopt")
        with self.assertRaisesRegex(ValueError, "unknown parallel"):
            TrainConfig(parallel="tp")
        with self.assertRaisesRegex(ValueError, "unknown distill loss"):
            DistillConfig(loss="mse")
        with self.assertRaisesRegex(ValueError, "unknown teacher_device"):
            DistillConfig(teacher_device="auto")
        with self.assertRaisesRegex(ValueError, "plain CE"):
            DistillConfig(loss="ce")
        with self.assertRaisesRegex(ValueError, "unsupported LoRA export"):
            ExportConfig(lora="both")

    def test_teacher_module_is_accepted(self):
        teacher = tiny_model()
        config = DistillConfig(teacher=teacher, teacher_device="cpu", top_k=16)
        self.assertIs(config.teacher, teacher)


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


class TestDistillModes(unittest.TestCase):
    def _trainer(self, distill_config):
        from transformers import TrainingArguments

        student = tiny_model()
        teacher = tiny_model()
        args = TrainingArguments(output_dir=tempfile.mkdtemp(), report_to=[], use_cpu=True)
        return QADTrainer(model=student, teacher=teacher, args=args, distill_config=distill_config), student

    def test_reverse_is_forwarded(self):
        trainer, student = self._trainer(DistillConfig(reverse=True, top_k=16, chunk=4))
        batch = {
            "input_ids": torch.randint(0, 128, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        loss = trainer.compute_loss(student, batch)
        with torch.no_grad():
            student_logits = student.logits(student(batch["input_ids"], batch["attention_mask"]))
            teacher_logits = trainer.teacher.logits(trainer.teacher(batch["input_ids"], batch["attention_mask"]))
            expected = topk_kl(student_logits, teacher_logits, k=16, reverse=True).mean()
        torch.testing.assert_close(loss, expected)

    def test_full_vocab_kl(self):
        trainer, student = self._trainer(DistillConfig(loss="kl", top_k=16, chunk=4))
        batch = {
            "input_ids": torch.randint(0, 128, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        loss = trainer.compute_loss(student, batch)
        with torch.no_grad():
            student_logits = student.logits(student(batch["input_ids"], batch["attention_mask"])).float()
            teacher_logits = trainer.teacher.logits(trainer.teacher(batch["input_ids"], batch["attention_mask"])).float()
            expected = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.log_softmax(teacher_logits, dim=-1),
                reduction="none",
                log_target=True,
            ).sum(dim=-1).mean()
        torch.testing.assert_close(loss, expected)

    def test_plain_ce(self):
        trainer, student = self._trainer(DistillConfig(loss="ce", kd_weight=0.0, ce_weight=1.0))
        batch = {
            "input_ids": torch.randint(0, 128, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
            "labels": torch.randint(0, 128, (1, 16)),
        }
        loss = trainer.compute_loss(student, batch)
        with torch.no_grad():
            logits = student.logits(student(batch["input_ids"], batch["attention_mask"]))
            expected = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(), batch["labels"][:, 1:].reshape(-1)
            )
        torch.testing.assert_close(loss, expected)


class TestLoraPaths(unittest.TestCase):
    def _model(self):
        cfg = FakeQuantConfig(
            rules=[QuantRule(module=r".*self_attn.q_proj$", weight="mxfp4", input="mxfp4", output="fp8_e4m3")]
        )
        return quantize(tiny_model(), cfg)

    def test_path_a_initializes_and_backwards(self):
        torch.manual_seed(0)
        model = apply_lora(
            self._model(),
            {"rank": 4, "alpha": 4, "targets": ["q_proj"], "quant_order": "quant_then_lora"},
        )
        self.assertEqual(model.base_model_prefix, "model")
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        output = layer(torch.randn(3, 64))
        output.sum().backward()
        self.assertIsNotNone(layer.lora_A.default.weight.grad)
        self.assertIsNotNone(layer.lora_B.default.weight.grad)

    def test_path_b_runs_output_quantizer(self):
        torch.manual_seed(0)
        model = apply_lora(
            self._model(),
            {"rank": 4, "alpha": 4, "targets": ["q_proj"], "quant_order": "lora_then_quant"},
        )
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        calls = []
        handle = layer.base_layer.output_quantizer.register_forward_hook(lambda *args: calls.append(1))
        layer(torch.randn(3, 64)).sum().backward()
        handle.remove()
        self.assertEqual(len(calls), 1)


class TestQuantMetadataAndExport(unittest.TestCase):
    def test_write_quantization_config(self):
        model = quantize(tiny_model(), tiny_quant_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = write_quantization_config(model, tmp, tiny_quant_config())
            payload = json.loads(Path(path).read_text())
        self.assertEqual(payload["schema"], "neural_compressor.qad/v1")
        self.assertEqual(payload["mode"], "fake_quant")
        self.assertIn("rules", payload["quant"])

    def test_export_text_roundtrip_excludes_quantizer_buffers(self):
        cfg = FakeQuantConfig(
            rules=[QuantRule(module=r".*self_attn.q_proj$", weight=dict(format="fp8_e4m3", scheme="static", amax=1.25))]
        )
        model = quantize(tiny_model(), cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out = export(model, None, ExportConfig(keep_vision=False), tmp, dtype=torch.float32)
            from safetensors import safe_open

            with safe_open(str(Path(out) / "model.safetensors"), "pt") as handle:
                self.assertFalse(any("quantizer" in key for key in handle.keys()))
            reloaded = quantize(tiny_model(), cfg)
            load_state_dict_into_model(reloaded, Path(out) / "model.safetensors")
            for key, value in reloaded.state_dict().items():
                if "quantizer" not in key:
                    torch.testing.assert_close(value, model.state_dict()[key])
            files = {item.name for item in Path(out).iterdir()}
            self.assertIn("config.json", files)

    def test_export_text_honors_shard_size_and_hf_reload(self):
        model = tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            export(model, None, ExportConfig(keep_vision=False, max_shard_size="1KB"), tmp, dtype=torch.float32)
            self.assertTrue((Path(tmp) / "model.safetensors.index.json").is_file())
            from transformers import AutoModelForCausalLM

            reloaded = AutoModelForCausalLM.from_pretrained(tmp, dtype=torch.float32)
            self.assertEqual(reloaded.state_dict().keys(), model.state_dict().keys())
            for key, value in reloaded.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key])

    def test_export_adapter_and_reload(self):
        model = apply_lora(
            quantize(tiny_model(), tiny_quant_config()),
            {"rank": 4, "alpha": 4, "targets": ["q_proj"], "quant_order": "quant_then_lora"},
        )
        expected = {name: value.detach().clone() for name, value in model.state_dict().items() if "lora_" in name}
        with tempfile.TemporaryDirectory() as tmp:
            export(model, None, ExportConfig(keep_vision=False, lora="adapter"), tmp)
            self.assertTrue((Path(tmp) / "adapter_model.safetensors").is_file())
            reloaded = apply_lora(
                quantize(tiny_model(), tiny_quant_config()),
                {"rank": 4, "alpha": 4, "targets": ["q_proj"], "quant_order": "quant_then_lora"},
            )
            reloaded.load_adapter(tmp, adapter_name="default", autocast_adapter_dtype=False)
            for name, value in expected.items():
                torch.testing.assert_close(reloaded.state_dict()[name], value)


class TestReviewFixes(unittest.TestCase):
    def test_hidden_uses_original_wrapper(self):
        student = tiny_model()

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.module = model
                self.calls = 0

            def forward(self, *args, **kwargs):
                self.calls += 1
                return self.module(*args, **kwargs)

        from transformers import TrainingArguments

        wrapped = Wrapper(student)
        trainer = QADTrainer(
            model=wrapped,
            teacher=tiny_model(),
            args=TrainingArguments(output_dir=tempfile.mkdtemp(), report_to=[], use_cpu=True),
            distill_config=DistillConfig(top_k=16),
        )
        trainer._hidden(wrapped, torch.randint(0, 128, (1, 4)), None)
        self.assertEqual(wrapped.calls, 1)

    def test_teacher_is_frozen_and_gets_no_gradients(self):
        from transformers import TrainingArguments

        student = tiny_model()
        teacher = tiny_model()
        trainer = QADTrainer(
            model=student,
            teacher=teacher,
            args=TrainingArguments(output_dir=tempfile.mkdtemp(), report_to=[], use_cpu=True),
            distill_config=DistillConfig(top_k=16, chunk=4),
        )
        self.assertFalse(teacher.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))
        batch = {
            "input_ids": torch.randint(0, 128, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        trainer.compute_loss(student, batch).backward()
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_save_adapter_keeps_trained_weights(self):
        from neural_compressor.torch.qad.checkpoint import save_adapter
        from safetensors import safe_open

        model = apply_lora(
            quantize(tiny_model(), tiny_quant_config()),
            {"rank": 4, "alpha": 4, "targets": ["q_proj"], "quant_order": "quant_then_lora"},
        )
        expected = {name: value.detach().clone() for name, value in model.state_dict().items() if "lora_" in name}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(save_adapter(model, tmp, should_save=True))
            reloaded = apply_lora(
                quantize(tiny_model(), tiny_quant_config()),
                {"rank": 4, "alpha": 4, "targets": ["q_proj"], "quant_order": "quant_then_lora"},
            )
            reloaded.load_adapter(tmp, adapter_name="default", autocast_adapter_dtype=False)
            for name, value in expected.items():
                torch.testing.assert_close(reloaded.state_dict()[name], value)

    def test_plain_model_load_skips_quantizer_metadata(self):
        cfg = FakeQuantConfig(
            rules=[QuantRule(module=r".*self_attn.q_proj$", weight=dict(format="fp8_e4m3", scheme="static", amax=1.25))]
        )
        model = quantize(tiny_model(), cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out = export(model, None, ExportConfig(keep_vision=False), tmp, dtype=torch.float32)
            plain = tiny_model()
            load_state_dict_into_model(plain, str(Path(out) / "model.safetensors"))
            for key, value in plain.state_dict().items():
                if "quantizer" not in key:
                    torch.testing.assert_close(value, model.state_dict()[key])

    def test_unsloth_source_is_not_registered_as_child(self):
        from neural_compressor.torch.qad.backends import wrap_unsloth

        source = tiny_model()
        wrapped = wrap_unsloth(source)
        self.assertFalse(any(name == "_unsloth_source" for name, _ in wrapped.named_modules()))

    def test_tiny_calibration_uses_synthetic_batches(self):
        from types import SimpleNamespace

        main = load_qad_example("main.py")
        recipe = Path(__file__).resolve().parents[3] / "examples/pytorch/llm/qwen/qad/recipes/qwen3_8_27b_mxfp4_mlp_fp8_attn.yaml"
        args = SimpleNamespace(
            recipe=str(recipe),
            model=None,
            teacher=None,
            amax_source=None,
            save_steps=None,
            seq_length=None,
            train_iters=None,
            top_k=None,
            output=tempfile.mkdtemp(),
            tiny=True,
            synthetic_batches=2,
            calib_batches=1,
            data="",
        )
        config = main.build_config(args)
        student = main.quantize_student(config, args, local_rank=0, world=1, device=torch.device("cpu"))
        self.assertIsNone(config.quant.amax_source)
        self.assertTrue(any(module.amax is not None for module in student.modules() if hasattr(module, "amax")))

    def test_prepare_data_writes_independent_document_count(self):
        from types import SimpleNamespace

        prepare_data = load_qad_example("prepare_data.py")

        class Tokenizer:
            eos_token_id = 99

            def apply_chat_template(self, *args, **kwargs):
                return "conversation"

            def __call__(self, text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[1, 2])

        import sys

        old_from_pretrained = prepare_data.AutoTokenizer.from_pretrained
        old_datasets = sys.modules.get("datasets")
        prepare_data.AutoTokenizer.from_pretrained = lambda path: Tokenizer()
        sys.modules["datasets"] = SimpleNamespace(load_dataset=lambda *args, **kwargs: iter([
            {"messages": [{"role": "user", "content": "a"}]},
            {"messages": [{"role": "user", "content": "b"}]},
        ]))
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "tokens"
            old_argv = sys.argv
            try:
                sys.argv = ["prepare_data.py", "--model", "fake", "--out", str(prefix), "--max-samples", "2"]
                prepare_data.main()
            finally:
                sys.argv = old_argv
                prepare_data.AutoTokenizer.from_pretrained = old_from_pretrained
                if old_datasets is None:
                    sys.modules.pop("datasets", None)
                else:
                    sys.modules["datasets"] = old_datasets
            with open(f"{prefix}.idx", "rb") as handle:
                self.assertEqual(handle.read(9), b"MMIDIDX\x00\x00")
                handle.read(8)
                handle.read(1)
                sequence_count = struct.unpack("<Q", handle.read(8))[0]
                document_count = struct.unpack("<Q", handle.read(8))[0]
                handle.read(sequence_count * 4)
                handle.read(sequence_count * 8)
                indices = np.frombuffer(handle.read(document_count * 8), dtype=np.int64)
        self.assertEqual(sequence_count, 2)
        self.assertEqual(document_count, 3)
        self.assertEqual(indices.tolist(), [0, 3, 6])


class TestSecondReviewFixes(unittest.TestCase):
    def test_tied_weights_export_to_independent_safetensors(self):
        from safetensors import safe_open

        model = tiny_model()
        model.lm_head.weight = model.model.embed_tokens.weight
        with tempfile.TemporaryDirectory() as tmp:
            out = export(model, None, ExportConfig(keep_vision=False), tmp, dtype=torch.float32)
            with safe_open(str(Path(out) / "model.safetensors"), "pt") as handle:
                embedding = handle.get_tensor("model.embed_tokens.weight")
                head = handle.get_tensor("lm_head.weight")
        torch.testing.assert_close(head, embedding)

    def test_lora_forwards_mixed_batch_adapter_names(self):
        from unittest.mock import patch

        from peft.tuners.lora.layer import Linear as LoraLinear

        model = apply_lora(
            quantize(tiny_model(), tiny_quant_config()),
            {"rank": 4, "alpha": 4, "targets": ["q_proj"], "quant_order": "lora_then_quant"},
        )
        layer = next(module for module in model.modules() if isinstance(module, LoraLinear))
        with patch.object(LoraLinear, "forward", autospec=True, return_value="output") as forward:
            result = layer(torch.randn(2, 4, 64), adapter_names=["default", "other"])
        self.assertEqual(result, "output")
        self.assertEqual(forward.call_args.kwargs["adapter_names"], ["default", "other"])

    def test_qwen_hub_id_resolves_to_snapshot(self):
        from unittest.mock import patch

        from neural_compressor.torch.qad.modeling import _resolve_checkpoint_dir

        with tempfile.TemporaryDirectory() as tmp:
            with patch("huggingface_hub.snapshot_download", return_value=tmp) as download:
                self.assertEqual(_resolve_checkpoint_dir("org/model"), Path(tmp))
                self.assertEqual(_resolve_checkpoint_dir(tmp), Path(tmp))
        download.assert_called_once_with(
            "org/model", allow_patterns=["*.safetensors", "model.safetensors.index.json"]
        )

    def test_generic_hf_hidden_uses_root_wrapper_output(self):
        from types import SimpleNamespace

        from transformers import TrainingArguments

        class GenericHF(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type="generic")
                self.model = torch.nn.Embedding(128, 16)
                self.lm_head = torch.nn.Linear(16, 128, bias=False)

            def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
                hidden = self.model(input_ids)
                return SimpleNamespace(hidden_states=(hidden,), logits=self.lm_head(hidden))

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.module = model
                self.calls = 0

            def forward(self, *args, **kwargs):
                self.calls += 1
                return self.module(*args, **kwargs)

        student = GenericHF()
        wrapped = Wrapper(student)
        trainer = QADTrainer(
            model=wrapped,
            teacher=tiny_model(),
            args=TrainingArguments(output_dir=tempfile.mkdtemp(), report_to=[], use_cpu=True),
            distill_config=DistillConfig(top_k=16),
        )
        hidden = trainer._hidden(wrapped, torch.randint(0, 128, (2, 4)), None)
        logits = trainer._logits(wrapped, hidden)
        self.assertEqual(wrapped.calls, 1)
        self.assertEqual(hidden.shape, (2, 4, 16))
        self.assertEqual(logits.shape, (2, 4, 128))

    def test_mixed_ce_requires_labels(self):
        from transformers import TrainingArguments

        student = tiny_model()
        trainer = QADTrainer(
            model=student,
            teacher=tiny_model(),
            args=TrainingArguments(output_dir=tempfile.mkdtemp(), report_to=[], use_cpu=True),
            distill_config=DistillConfig(top_k=16, kd_weight=0.0, ce_weight=1.0),
        )
        batch = {
            "input_ids": torch.randint(0, 128, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        with self.assertRaisesRegex(ValueError, "ce_weight > 0 requires labels"):
            trainer.compute_loss(student, batch)
        batch["labels"] = torch.randint(0, 128, (1, 16))
        self.assertTrue(torch.isfinite(trainer.compute_loss(student, batch)))


if __name__ == "__main__":
    unittest.main()
