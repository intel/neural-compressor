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
"""Unit tests for PEFT QuantLoraLinear plugin and standalone QATTrainer."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments

from neural_compressor.torch.quantization import FakeQuantConfig, QATTrainer, QuantSchemeRule, prepare_qat
from neural_compressor.torch.quantization.fake_quant import (
    QuantLinear,
    QuantLoraLinear,
    convert_peft_lora_to_quant_lora,
    disable_fake_quant,
    ensure_peft_torchao_compat,
    load_quantizer_state,
    merge_lora_into_quant_linear,
)


def _tiny_qwen2(dtype=torch.float32):
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return Qwen2ForCausalLM(cfg).to(dtype=dtype)


class TestPeftQuantLoraPlugin(unittest.TestCase):
    def test_both_workflow_orders_match_and_avoid_nested_quantizers(self):
        ensure_peft_torchao_compat()
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(42)
        base1 = _tiny_qwen2()
        base2 = _tiny_qwen2()
        base2.load_state_dict(base1.state_dict())

        lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], bias="none")
        qcfg = FakeQuantConfig(
            config_groups=[
                QuantSchemeRule(targets=["q_proj", "v_proj", "k_proj"], weights="mxfp4", input_activations="mxfp4")
            ],
            ignore=["lm_head"],
        )

        # Workflow 1: get_peft_model -> prepare_qat
        torch.manual_seed(100)
        m1 = get_peft_model(base1, lora_cfg)
        prepare_qat(m1, qcfg)

        # Workflow 2: prepare_qat -> get_peft_model -> convert_peft_lora_to_quant_lora
        prepare_qat(base2, qcfg)
        torch.manual_seed(100)
        m2 = get_peft_model(base2, lora_cfg)
        converted = convert_peft_lora_to_quant_lora(m2, quant_order="lora_then_quant")
        self.assertEqual(converted, 4)

        for model in (m1, m2):
            qlora_layers = [m for m in model.modules() if isinstance(m, QuantLoraLinear)]
            self.assertEqual(len(qlora_layers), 4)
            for layer in qlora_layers:
                self.assertIsInstance(layer.base_layer, nn.Linear)
                self.assertNotIsInstance(layer.base_layer, QuantLinear)
                with torch.no_grad():
                    nn.init.ones_(layer.lora_B["default"].weight)

        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            with torch.no_grad():
                p2.copy_(p1)

        ids = torch.randint(0, 128, (2, 8))
        out1 = m1(ids).logits
        out2 = m2(ids).logits
        self.assertTrue(torch.equal(out1, out2))

    def test_self_teacher_and_merge_lora_into_quant_linear_exact(self):
        ensure_peft_torchao_compat()
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(7)
        float_ref = _tiny_qwen2().eval()
        student = _tiny_qwen2()
        student.load_state_dict(float_ref.state_dict())

        peft_model = get_peft_model(
            student, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "up_proj"], bias="none")
        )
        prepare_qat(peft_model, FakeQuantConfig.mxfp4())
        for mod in peft_model.modules():
            if isinstance(mod, QuantLoraLinear):
                with torch.no_grad():
                    mod.lora_B["default"].weight.normal_(0, 0.05)

        peft_model.eval()
        ids = torch.randint(0, 128, (2, 8))

        # 1. Self-teacher toggle recovers exact float reference output
        with torch.no_grad(), disable_fake_quant(peft_model), peft_model.disable_adapter():
            self_teacher_logits = peft_model(ids).logits
        with torch.no_grad():
            ref_logits = float_ref(ids).logits
        self.assertTrue(torch.equal(self_teacher_logits, ref_logits))

        # 2. merge_lora_into_quant_linear preserves exact Path-B quantized output
        with torch.no_grad():
            before_merge = peft_model(ids).logits
        merged = merge_lora_into_quant_linear(peft_model).eval()
        self.assertFalse(any(isinstance(m, QuantLoraLinear) for m in merged.modules()))
        self.assertTrue(any(isinstance(m, QuantLinear) for m in merged.modules()))
        with torch.no_grad():
            after_merge = merged(ids).logits
        torch.testing.assert_close(before_merge, after_merge, rtol=0.0, atol=0.0)


class TestStandaloneQATTrainer(unittest.TestCase):
    def test_auto_calibration_training_and_lossless_state_save(self):
        torch.manual_seed(11)
        model = _tiny_qwen2()
        qcfg = FakeQuantConfig(
            config_groups=[
                QuantSchemeRule(
                    targets=["q_proj", "k_proj"],
                    weights={"format": "fp8_e4m3", "dynamic": False},
                    input_activations={"format": "fp8_e4m3", "dynamic": False},
                )
            ],
            ignore=["lm_head"],
        )
        dataset = [
            {
                "input_ids": torch.randint(0, 128, (12,)),
                "attention_mask": torch.ones(12, dtype=torch.long),
                "labels": torch.randint(0, 128, (12,)),
            }
            for _ in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                per_device_train_batch_size=2,
                max_steps=2,
                learning_rate=1e-4,
                save_strategy="no",
                report_to=[],
                use_cpu=True,
                remove_unused_columns=False,
            )
            trainer = QATTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                quant_config=qcfg,
                calib_batches=2,
            )
            trainer.train()
            save_dir = Path(tmp) / "final"
            trainer.save_model(str(save_dir))
            self.assertTrue((save_dir / "quantizer_state.json").is_file())
            self.assertTrue((save_dir / "quant_config.json").is_file())

            fresh = _tiny_qwen2()
            prepare_qat(fresh, qcfg)
            restored = load_quantizer_state(fresh, save_dir / "quantizer_state.json")
            self.assertEqual(len(restored["loaded"]), 8)  # 2 layers * 2 projs * (weight + input)
            self.assertEqual(restored["missing"], [])


if __name__ == "__main__":
    unittest.main()
