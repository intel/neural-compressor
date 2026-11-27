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
"""Unit tests for v2.0 QAD (QADTrainer, QADTrainingArguments, prepare_qad_model, and Qwen QAD example helpers)."""

import importlib.util
import json
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import TrainingArguments

from neural_compressor.torch.distillation import (
    DistillConfig,
    DistillationConfig,
    DistillationTrainingArguments,
    LogitsKLLoss,
    TopKLogitsKLLoss,
    topk_kl,
)
from neural_compressor.torch.export import export_hf_checkpoint
from neural_compressor.torch.qad import (
    QADTrainer,
    QADTrainingArguments,
    decoder_layer_class_names,
    prepare_qad_model,
    teacher_placement,
)
from neural_compressor.torch.quantization import (
    FakeQuantConfig,
    QuantLoraLinear,
    QuantSchemeRule,
    QuantizationTrainingArguments,
    prepare_qat,
    quantize,
    save_fsdp2_peft_adapter,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples/pytorch/llm/qwen/qad"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


def load_qad_example(rel_path: str):
    path = EXAMPLE_DIR / rel_path
    mod_name = "qad_example_" + rel_path.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def tiny_model(dtype=torch.float32, layer_types=None):
    loader = load_qad_example("modeling/qwen35_text_loader.py")
    return loader.build_tiny_qwen35(
        num_hidden_layers=2,
        layer_types=layer_types or ["full_attention", "full_attention"],
        dtype=dtype,
    )


def tiny_quant_config():
    return FakeQuantConfig(
        config_groups=[
            QuantSchemeRule(
                targets=[r"re:.*\.mlp\.(gate_proj|up_proj|down_proj)$"],
                weights="mxfp4",
                input_activations="mxfp4",
            ),
            QuantSchemeRule(
                targets=[r"re:.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"],
                weights="fp8_e4m3",
            ),
        ]
    )


def _apply_peft_lora(model, *, rank=4, alpha=4, targets=None, merge_before_quant=True):
    return prepare_qad_model(
        model,
        tiny_quant_config(),
        lora_config={
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": targets or ["q_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        lora_merge_before_quant=merge_before_quant,
    )


class TestPackedDataset(unittest.TestCase):
    def test_tail_mask_excludes_padding(self):
        ds_mod = load_qad_example("data/megatron_indexed_dataset.py")
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
            sample = ds_mod.PackedDataset(str(prefix), 4, drop_tail=False)[2]
        self.assertEqual(sample["input_ids"].tolist(), [8, 9, 0, 0])
        self.assertEqual(sample["attention_mask"].tolist(), [1, 1, 0, 0])
        self.assertEqual(sample["loss_mask"].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(sample["labels"].tolist(), [8, 9, -100, -100])


def modelopt_topk_reference(student, teacher, k, temperature, reverse):
    teacher_values, teacher_indices = torch.topk(teacher, k=k, dim=-1)
    student_values = torch.gather(student, dim=-1, index=teacher_indices)
    student_log_probs = F.log_softmax(student_values.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_values.float() / temperature, dim=-1)
    if reverse:
        student_log_probs, teacher_log_probs = teacher_log_probs, student_log_probs
    return F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)


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
        torch.testing.assert_close(LogitsKLLoss()(student, teacher), reference)

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

    def test_reverse_matches_modelopt_formula_and_gradient(self):
        torch.manual_seed(7)
        base_student = torch.randn(2, 5, 8)
        teacher = torch.randn(2, 5, 8)
        for reverse in (False, True):
            for temperature in (1.0, 2.0):
                student = base_student.clone().requires_grad_(True)
                expected_student = base_student.clone().requires_grad_(True)
                actual = topk_kl(student, teacher, k=2, temperature=temperature, reverse=reverse)
                expected = modelopt_topk_reference(
                    expected_student, teacher, k=2, temperature=temperature, reverse=reverse
                )
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(student.grad, expected_student.grad, rtol=0.0, atol=0.0)

    def test_reverse_precomputed_topk_matches_tensor_path(self):
        torch.manual_seed(8)
        student = torch.randn(2, 5, 8)
        teacher = torch.randn(2, 5, 8)
        criterion = TopKLogitsKLLoss(top_k=2, temperature=1.0, chunk=3, reverse=True)
        values, indices = torch.topk(teacher, k=2, dim=-1)
        tensor_loss, tensor_count = criterion(student, teacher)
        tuple_loss, tuple_count = criterion(student, (values, indices))
        torch.testing.assert_close(tuple_loss, tensor_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(tuple_count, tensor_count)
        expected = modelopt_topk_reference(student, teacher, k=2, temperature=1.0, reverse=True).sum()
        torch.testing.assert_close(tuple_loss, expected, rtol=0.0, atol=0.0)


class TestParallelAndArguments(unittest.TestCase):
    def test_decoder_layer_class_names_on_standard_and_peft_models(self):
        model = tiny_model()
        self.assertEqual(decoder_layer_class_names(model), ["Qwen3_5DecoderLayer"])
        lora_model = _apply_peft_lora(tiny_model())
        self.assertEqual(decoder_layer_class_names(lora_model), ["Qwen3_5DecoderLayer"])
        self.assertIn(teacher_placement(model), ("cuda", "cpu", "shard"))

    def test_qad_training_arguments_and_yaml_recipes(self):
        self.assertTrue(issubclass(QADTrainingArguments, DistillationTrainingArguments))
        self.assertTrue(issubclass(QADTrainingArguments, QuantizationTrainingArguments))

        with tempfile.TemporaryDirectory() as tmp:
            qad_args = QADTrainingArguments(
                output_dir=tmp,
                teacher_model_name_or_path="self",
                teacher_placement="self",
                kd_top_k=64,
                logits_chunk_size=32,
                use_cpu=True,
                bf16=False,
            )
            d_cfg = qad_args.build_distill_config()
            self.assertEqual(d_cfg.teacher, "self")
            self.assertEqual(d_cfg.top_k, 64)
            self.assertEqual(d_cfg.chunk_size, 32)

        for recipe_name in (
            "qwen3_5_27b_mxfp4_fp8_full_fsdp2.yaml",
            "qwen3_5_27b_mxfp4_fp8_full_cpu_offload.yaml",
            "qwen3_5_27b_mxfp4_fp8_lora_path_b.yaml",
            "qwen3_5_27b_nvfp4_fp8_full_fsdp2.yaml",
            "qwen3_8_27b_mxfp4_mlp_fp8_attn.yaml",
        ):
            cfg = FakeQuantConfig.from_yaml(EXAMPLE_DIR / "recipes" / recipe_name)
            self.assertGreaterEqual(len(cfg.config_groups), 2)

    def test_invalid_distill_choices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown distill loss"):
            DistillationConfig(loss="mse")
        with self.assertRaisesRegex(ValueError, "unknown teacher_device"):
            DistillationConfig(teacher_device="tp")
        with self.assertRaisesRegex(ValueError, "plain CE"):
            DistillationConfig(loss="ce")

    def test_teacher_module_is_accepted(self):
        teacher = tiny_model()
        config = DistillationConfig(teacher=teacher, teacher_device="cpu", top_k=16)
        self.assertIs(config.teacher, teacher)


class TestQADTrainer(unittest.TestCase):
    def _trainer(self, student, teacher):
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
        teacher = tiny_model().eval()
        student = quantize(tiny_model(), tiny_quant_config())
        trainer = self._trainer(student, teacher)

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "loss_mask": torch.ones(2, 16),
        }
        loss = trainer.compute_loss(student, batch)

        with torch.no_grad():
            student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            teacher_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
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
            student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            teacher_logits = trainer.teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
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
            student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits.float()
            teacher_logits = trainer.teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits.float()
            expected = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.log_softmax(teacher_logits, dim=-1),
                reduction="none",
                log_target=True,
            ).sum(dim=-1).mean()
        torch.testing.assert_close(loss, expected)

    def test_full_vocab_reverse_matches_manual_formula(self):
        trainer, student = self._trainer(DistillConfig(loss="kl", reverse=True, top_k=16, chunk=4))
        batch = {
            "input_ids": torch.randint(0, 128, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        loss = trainer.compute_loss(student, batch)
        with torch.no_grad():
            student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits.float()
            teacher_logits = trainer.teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits.float()
            expected = F.kl_div(
                F.log_softmax(teacher_logits, dim=-1),
                F.log_softmax(student_logits, dim=-1),
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
            logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            expected = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(), batch["labels"][:, 1:].reshape(-1)
            )
        torch.testing.assert_close(loss, expected)


class TestLoraAndExportRoundtrip(unittest.TestCase):
    def _lora_with_output_quant(self, merge_before_quant: bool):
        cfg = FakeQuantConfig(
            config_groups=[
                QuantSchemeRule(
                    targets=[r"re:.*self_attn\.q_proj$"],
                    weights="mxfp4",
                    input_activations="mxfp4",
                    output_activations="fp8_e4m3",
                )
            ]
        )
        return prepare_qad_model(
            tiny_model(),
            cfg,
            lora_config={"r": 4, "lora_alpha": 4, "target_modules": ["q_proj"], "bias": "none", "task_type": "CAUSAL_LM"},
            lora_merge_before_quant=merge_before_quant,
        )

    def test_path_a_initializes_and_backwards(self):
        torch.manual_seed(0)
        model = self._lora_with_output_quant(merge_before_quant=False)
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        self.assertIsInstance(layer, QuantLoraLinear)
        output = layer(torch.randn(3, 64))
        output.sum().backward()
        self.assertIsNotNone(layer.lora_A.default.weight.grad)
        self.assertIsNotNone(layer.lora_B.default.weight.grad)

    def test_path_b_runs_output_quantizer(self):
        torch.manual_seed(0)
        model = self._lora_with_output_quant(merge_before_quant=True)
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        self.assertIsInstance(layer, QuantLoraLinear)
        calls = []
        handle = layer.output_quantizer.register_forward_hook(lambda *args: calls.append(1))
        layer(torch.randn(3, 64)).sum().backward()
        handle.remove()
        self.assertEqual(len(calls), 1)

    def test_export_hf_checkpoint_excludes_quantizer_buffers(self):
        cfg = FakeQuantConfig(
            config_groups=[
                QuantSchemeRule(
                    targets=[r"re:.*self_attn\.q_proj$"],
                    weights=dict(format="fp8_e4m3", dynamic=False, amax=1.25),
                )
            ]
        )
        model = tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            out_dir = Path(tmp) / "exported"
            model.save_pretrained(ckpt_dir)
            prepare_qat(model, cfg)
            from neural_compressor.torch.quantization import save_quantizer_state

            save_quantizer_state(model, ckpt_dir / "quantizer_state.json")
            cfg.to_json_file(ckpt_dir / "quant_config.json")
            export_hf_checkpoint(ckpt_dir, out_dir, export_format="bf16")
            from safetensors import safe_open

            with safe_open(str(out_dir / "model.safetensors"), "pt") as handle:
                self.assertFalse(any("quantizer" in key for key in handle.keys()))
            self.assertTrue((out_dir / "config.json").is_file())

    def test_lora_forwards_mixed_batch_adapter_names(self):
        from peft.tuners.lora.layer import Linear as LoraLinear

        model = _apply_peft_lora(tiny_model(), merge_before_quant=True)
        layer = next(module for module in model.modules() if isinstance(module, LoraLinear))
        with patch.object(LoraLinear, "forward", autospec=True, return_value="output") as forward:
            result = layer(torch.randn(2, 4, 64), adapter_names=["default", "other"])
        self.assertEqual(result, "output")
        self.assertEqual(forward.call_args.kwargs["adapter_names"], ["default", "other"])


class TestWrapperAndExampleHelpers(unittest.TestCase):
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
        model = _apply_peft_lora(tiny_model(), merge_before_quant=False)
        expected = {name: value.detach().clone() for name, value in model.state_dict().items() if "lora_" in name}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(save_fsdp2_peft_adapter(model, tmp, should_save=True))
            reloaded = _apply_peft_lora(tiny_model(), merge_before_quant=False)
            reloaded.load_adapter(tmp, adapter_name="default", autocast_adapter_dtype=False)
            for name, value in expected.items():
                torch.testing.assert_close(reloaded.state_dict()[name], value)

    def test_qwen_hub_id_resolves_to_snapshot(self):
        loader = load_qad_example("modeling/qwen35_text_loader.py")
        with tempfile.TemporaryDirectory() as tmp:
            with patch("huggingface_hub.snapshot_download", return_value=tmp) as download:
                self.assertEqual(loader._resolve_checkpoint_dir("org/model"), Path(tmp))
                self.assertEqual(loader._resolve_checkpoint_dir(tmp), Path(tmp))
        download.assert_called_once()

    def test_train_script_tiny_smoke_calibrates_and_runs(self):
        train_mod = load_qad_example("train.py")
        recipe = EXAMPLE_DIR / "recipes/qwen3_5_27b_mxfp4_fp8_lora_path_b.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            trainer = train_mod.run_training(
                [
                    "--config",
                    str(recipe),
                    "--tiny",
                    "--synthetic_batches",
                    "4",
                    "--calib_batches",
                    "1",
                    "--seq_length",
                    "32",
                    "--kd_top_k",
                    "16",
                    "--max_steps",
                    "1",
                    "--gradient_accumulation_steps",
                    "1",
                    "--save_strategy",
                    "no",
                    "--use_cpu",
                    "true",
                    "--bf16",
                    "false",
                    "--output_dir",
                    tmp,
                ]
            )
            self.assertEqual(trainer.state.global_step, 1)
            self.assertTrue(any(getattr(m, "amax", None) is not None for m in trainer.model.modules()))

    def test_prepare_data_writes_independent_document_count(self):
        prepare_data = load_qad_example("prepare_data.py")

        class Tokenizer:
            eos_token_id = 99

            def apply_chat_template(self, *args, **kwargs):
                return "conversation"

            def __call__(self, text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[1, 2])

        old_from_pretrained = prepare_data.AutoTokenizer.from_pretrained
        old_datasets = sys.modules.get("datasets")
        prepare_data.AutoTokenizer.from_pretrained = lambda path: Tokenizer()
        sys.modules["datasets"] = SimpleNamespace(
            load_dataset=lambda *args, **kwargs: iter(
                [
                    {"messages": [{"role": "user", "content": "a"}]},
                    {"messages": [{"role": "user", "content": "b"}]},
                ]
            )
        )
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

    def test_generic_hf_hidden_uses_root_wrapper_output(self):
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
