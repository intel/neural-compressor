from math import isclose

import pytest
from transformers import AutoTokenizer


class TestTansformersLikeAPI:
    def test_quantization_for_llm(self):
        model_name_or_path = "hf-internal-testing/tiny-random-gptj"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        from neural_compressor.transformers import (
            AutoModelForCausalLM,
            AutoRoundConfig,
            AwqConfig,
            GPTQConfig,
            RtnConfig,
            TeqConfig,
        )

        fp32_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        dummy_input = fp32_model.dummy_inputs["input_ids"]

        # weight-only
        # RTN
        woq_config = RtnConfig(bits=4)
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=woq_config,
        )
        woq_model.eval()
        output = woq_model(dummy_input)
        assert isclose(float(output[0][0][0][0]), 0.17631684243679047, rel_tol=1e-04)

        # AWQ
        woq_config = AwqConfig(bits=4, zero_point=False, n_samples=5, batch_size=1, seq_len=512, tokenizer=tokenizer)

        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, quantization_config=woq_config, use_neural_speed=False
        )
        woq_model.eval()
        output = woq_model(dummy_input)
        assert isclose(float(output[0][0][0][0]), 0.20071472227573395, rel_tol=1e-04)

        # TEQ
        woq_config = TeqConfig(bits=4, n_samples=5, batch_size=1, seq_len=512, tokenizer=tokenizer)
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, quantization_config=woq_config, use_neural_speed=False
        )
        woq_model.eval()
        output = woq_model(dummy_input)
        assert isclose(float(output[0][0][0][0]), 0.17631684243679047, rel_tol=1e-04)

        # fp8
        woq_config = RtnConfig(bits=8, weight_dtype="fp8_e5m2", scale_dtype="fp8_e8m0")
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, quantization_config=woq_config, use_neural_speed=False
        )
        woq_model.eval()
        output = woq_model(dummy_input)
        assert isclose(float(output[0][0][0][0]), 0.16162332892417908, rel_tol=1e-04)

        # load_in_4bit
        bit4_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_4bit=True, use_neural_speed=False)
        bit4_model.eval()
        output = bit4_model(dummy_input)
        assert isclose(float(output[0][0][0][0]), 0.17631684243679047, rel_tol=1e-04)

        # load_in_8bit
        bit8_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, load_in_8bit=True, use_neural_speed=False, device_map="cpu"
        )
        bit8_model.eval()
        output = bit8_model(dummy_input)
        assert isclose(float(output[0][0][0][0]), 0.16759155690670013, rel_tol=1e-04)

        # GPTQ
        woq_config = GPTQConfig(
            bits=4,
            weight_dtype="int4_clip",
            sym=True,
            desc_act=False,
            damp_percent=0.01,
            blocksize=32,
            n_samples=3,
            seq_len=256,
            tokenizer=tokenizer,
            batch_size=1,
        )
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, quantization_config=woq_config, use_neural_speed=False
        )
        woq_model.eval()
        output = woq_model(dummy_input)
        assert isclose(float(output[0][0][0][0]), 0.1800851970911026, rel_tol=1e-04)

        # AUTOROUND
        woq_config = AutoRoundConfig(
            bits=4, weight_dtype="int4_clip", n_samples=128, seq_len=32, iters=5, tokenizer=tokenizer
        )
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, quantization_config=woq_config, use_neural_speed=False
        )
        woq_model.eval()
        output = woq_model(dummy_input)
