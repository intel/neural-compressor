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
"""Unit tests for standalone neural_compressor.torch.distillation."""

import copy
import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments

from neural_compressor.torch.distillation import (
    DistillationConfig,
    KDTrainer,
    PrecomputedTopKCollator,
    PrecomputedTopKDataset,
    extract_teacher_topk_batch,
    pack_sft_sequences,
    save_precomputed_topk_shard,
    topk_kl,
)
from neural_compressor.torch.quantization import FakeQuantConfig, prepare_qat
from neural_compressor.torch.algorithms.fake_quant import QuantLoraLinear, ensure_peft_torchao_compat


def _tiny_qwen2():
    cfg = Qwen2Config(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return Qwen2ForCausalLM(cfg)


class TestKDTrainerAndPacking(unittest.TestCase):
    def test_zero_full_logit_allocation_and_single_pass_hybrid_loss(self):
        torch.manual_seed(0)
        student = _tiny_qwen2()
        teacher = _tiny_qwen2().eval()

        head_shapes = []
        orig_head_forward = student.lm_head.forward

        def spy_head(x):
            head_shapes.append(tuple(x.shape))
            return orig_head_forward(x)

        student.lm_head.forward = spy_head

        backbone_calls = []
        orig_backbone = student.model.forward

        def spy_backbone(*args, **kwargs):
            backbone_calls.append(kwargs.get("output_hidden_states", False))
            return orig_backbone(*args, **kwargs)

        student.model.forward = spy_backbone

        with tempfile.TemporaryDirectory() as tmp:
            trainer = KDTrainer(
                model=student,
                teacher_model=teacher,
                args=TrainingArguments(output_dir=tmp, report_to=[], use_cpu=True),
                distill_config=DistillationConfig(top_k=16, chunk=5, kd_weight=0.7, ce_weight=0.3),
            )
            batch = {
                "input_ids": torch.randint(0, 96, (2, 12)),
                "attention_mask": torch.ones(2, 12, dtype=torch.long),
                "labels": torch.randint(0, 96, (2, 12)),
                "loss_mask": torch.ones(2, 12),
            }
            loss = trainer.compute_loss(student, batch)

        # Single student backbone pass, never output_hidden_states=True
        self.assertEqual(backbone_calls, [False])
        # lm_head only ever sees chunks of length <= 5, NEVER full seq_len=12!
        self.assertEqual(head_shapes, [(2, 5, 64), (2, 5, 64), (2, 2, 64)])

        # Verify numerical exactness against full reference
        student.lm_head.forward = orig_head_forward
        student.model.forward = orig_backbone
        with torch.no_grad():
            s_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            t_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            expected_kd = topk_kl(s_logits, t_logits, k=16).mean()
            expected_ce = F.cross_entropy(
                s_logits[:, :-1].reshape(-1, 96).float(),
                batch["labels"][:, 1:].reshape(-1),
            )
            expected = 0.7 * expected_kd + 0.3 * expected_ce
        torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)

    def test_self_teacher_mode_matches_external_float_teacher_exact(self):
        ensure_peft_torchao_compat()
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(3)
        base = _tiny_qwen2()
        external_teacher = copy.deepcopy(base).eval()

        peft_student = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
        prepare_qat(peft_student, FakeQuantConfig.mxfp4())
        for mod in peft_student.modules():
            if isinstance(mod, QuantLoraLinear):
                with torch.no_grad():
                    mod.lora_B["default"].weight.normal_(0, 0.05)

        peft_student_2 = copy.deepcopy(peft_student)
        batch = {
            "input_ids": torch.randint(0, 96, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as tmp:
            t_self = KDTrainer(
                model=peft_student,
                teacher_model="self",
                args=TrainingArguments(output_dir=tmp, report_to=[], use_cpu=True),
                distill_config=DistillationConfig(teacher="self", top_k=12, chunk=4),
            )
            t_ext = KDTrainer(
                model=peft_student_2,
                teacher_model=external_teacher,
                args=TrainingArguments(output_dir=tmp, report_to=[], use_cpu=True),
                distill_config=DistillationConfig(top_k=12, chunk=4),
            )
            loss_self = t_self.compute_loss(peft_student, batch)
            loss_ext = t_ext.compute_loss(peft_student_2, batch)
            torch.testing.assert_close(loss_self, loss_ext, rtol=0.0, atol=0.0)
            loss_self.backward()
            loss_ext.backward()

        for (n1, p1), (n2, p2) in zip(peft_student.named_parameters(), peft_student_2.named_parameters()):
            if p1.requires_grad:
                torch.testing.assert_close(p1.grad, p2.grad, rtol=0.0, atol=0.0)

    def test_offline_topk_dump_and_collator_match_online_exact(self):
        torch.manual_seed(5)
        student = _tiny_qwen2()
        teacher = _tiny_qwen2().eval()
        raw_samples = [
            {"input_ids": torch.randint(0, 96, (8,)), "attention_mask": torch.ones(8, dtype=torch.long)}
            for _ in range(2)
        ]
        shard_rows = []
        with torch.no_grad():
            for sample in raw_samples:
                t_logits = teacher(sample["input_ids"].unsqueeze(0), sample["attention_mask"].unsqueeze(0)).logits[0]
                vals, inds = extract_teacher_topk_batch(t_logits, top_k=10, values_dtype=torch.float32)
                shard_rows.append(
                    {
                        **sample,
                        "teacher_topk_values": vals,
                        "teacher_topk_indices": inds,
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            save_precomputed_topk_shard(shard_rows, f"{tmp}/shard_0.pt")
            ds = PrecomputedTopKDataset(tmp)
            collator = PrecomputedTopKCollator(pad_token_id=0)
            offline_batch = collator([ds[0], ds[1]])
            online_batch = {
                "input_ids": offline_batch["input_ids"],
                "attention_mask": offline_batch["attention_mask"],
            }
            t_off = KDTrainer(
                model=student,
                teacher_model=None,
                args=TrainingArguments(output_dir=tmp, report_to=[], use_cpu=True),
                distill_config=DistillationConfig(teacher=None, top_k=10, chunk=4),
            )
            t_on = KDTrainer(
                model=student,
                teacher_model=teacher,
                args=TrainingArguments(output_dir=tmp, report_to=[], use_cpu=True),
                distill_config=DistillationConfig(top_k=10, chunk=4),
            )
            loss_off = t_off.compute_loss(student, offline_batch)
            loss_on = t_on.compute_loss(student, online_batch)
            torch.testing.assert_close(loss_off, loss_on, rtol=0.0, atol=0.0)

    def test_pack_sft_sequences_resets_position_ids_per_document(self):
        seqs = [
            {"input_ids": [10, 11, 12], "loss_mask": [0, 1, 1]},
            {"input_ids": [20, 21], "loss_mask": [1, 1]},
        ]
        packed = pack_sft_sequences(seqs, seq_length=8, pad_token_id=0)
        self.assertEqual(len(packed), 1)
        row = packed[0]
        self.assertEqual(row["input_ids"].tolist(), [10, 11, 12, 20, 21, 0, 0, 0])
        self.assertEqual(row["position_ids"].tolist(), [0, 1, 2, 0, 1, 0, 1, 2])
        self.assertEqual(row["attention_mask"].tolist(), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(row["loss_mask"].tolist(), [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(row["labels"].tolist(), [-100, 11, 12, -100, 21, -100, -100, -100])


if __name__ == "__main__":
    unittest.main()
