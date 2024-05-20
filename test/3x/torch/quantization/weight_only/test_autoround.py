import copy
import shutil

import pytest
import torch
import transformers
from packaging.version import Version

from neural_compressor.torch.algorithms.weight_only.autoround import AutoRoundQuantizer, get_autoround_default_run_fn
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    convert,
    get_default_AutoRound_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import logger

try:
    import auto_round

    AUTO_ROUND_VERSION_0_11 = Version("0.11")

    auto_round_version = auto_round.__version__.split("+")[0]
    auto_round_version = Version(auto_round_version)
    auto_round_installed = True
except ImportError:
    auto_round_installed = False


@pytest.mark.skipif(not auto_round_installed, reason="auto_round module is not installed")
class TestAutoRound:
    def setup_class(self):
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
        )
        self.inp = torch.ones([1, 10], dtype=torch.long)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM", trust_remote_code=True
        )
        self.label = self.gptj(self.inp)[0]

    def teardown_class(self):
        shutil.rmtree("saved_results", ignore_errors=True)

    def setup_method(self, method):
        logger.info(f"Running TestAutoRound test: {method.__name__}")

    def test_autoround(self):
        gpt_j_model = copy.deepcopy(self.gptj)
        quant_config = AutoRoundConfig(n_samples=20, seqlen=10, iters=10, scale_dtype="fp32")
        logger.info(f"Test AutoRound with config {quant_config}")

        run_fn = get_autoround_default_run_fn
        run_args = (
            self.tokenizer,
            "NeelNanda/pile-10k",
            20,
            10,
        )
        fp32_model = gpt_j_model

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)
        run_fn(model, *run_args)
        q_model = convert(model)
        out = q_model(self.inp)[0]
        assert torch.allclose(out, self.label, atol=1e-1)
        assert "transformer.h.0.attn.k_proj" in q_model.autoround_config.keys()
        assert "scale" in q_model.autoround_config["transformer.h.0.attn.k_proj"].keys()
        assert torch.float32 == q_model.autoround_config["transformer.h.0.attn.k_proj"]["scale_dtype"]

    def test_quantizer(self):
        gpt_j_model = copy.deepcopy(self.gptj)

        run_fn = get_autoround_default_run_fn
        run_args = (
            self.tokenizer,
            "NeelNanda/pile-10k",
            20,
            10,
        )
        weight_config = {
            "*": {
                "data_type": "int",
                "bits": 4,
                "group_size": 32,
                "sym": False,
            }
        }
        quantizer = AutoRoundQuantizer(quant_config=weight_config)
        fp32_model = gpt_j_model

        # quantizer execute
        model = quantizer.prepare(model=fp32_model)
        run_fn(model, *run_args)
        q_model = quantizer.convert(model)

        out = q_model(self.inp)[0]
        assert torch.allclose(self.label, out, atol=1e-1)
        assert "transformer.h.0.attn.k_proj" in q_model.autoround_config.keys()
        assert "scale" in q_model.autoround_config["transformer.h.0.attn.k_proj"].keys()
        assert torch.float32 == q_model.autoround_config["transformer.h.0.attn.k_proj"]["scale_dtype"]

    def test_autoround_with_quantize_API(self):
        gpt_j_model = copy.deepcopy(self.gptj)

        quant_config = get_default_AutoRound_config()
        logger.info(f"Test AutoRound with config {quant_config}")

        # quantize API
        q_model = quantize(
            model=gpt_j_model,
            quant_config=quant_config,
            run_fn=get_autoround_default_run_fn,
            run_args=(
                self.tokenizer,
                "NeelNanda/pile-10k",
                20,
                10,
            ),
        )
        out = q_model(self.inp)[0]
        assert torch.allclose(out, self.label, atol=1e-1)

    def test_save_and_load(self):
        fp32_model = copy.deepcopy(self.gptj)
        quant_config = get_default_AutoRound_config()
        logger.info(f"Test AutoRound with config {quant_config}")

        run_fn = get_autoround_default_run_fn
        run_args = (
            self.tokenizer,
            "NeelNanda/pile-10k",
            20,
            10,
        )
        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        run_fn(model, *run_args)
        q_model = convert(model)

        assert q_model is not None, "Quantization failed!"
        q_model.save("saved_results")
        inc_out = q_model(self.inp)[0]

        from neural_compressor.torch.quantization import load

        # loading compressed model
        loaded_model = load("saved_results")
        loaded_out = loaded_model(self.inp)[0]
        assert torch.allclose(inc_out, loaded_out), "Unexpected result. Please double check."

    @pytest.mark.skipif(auto_round_version <= AUTO_ROUND_VERSION_0_11, reason="Requires auto_round>=0.11")
    def test_conv1d(self):
        input = torch.randn(1, 32)
        from transformers import GPT2Model, GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        out1 = model(**encoded_input)[0]
        run_fn = get_autoround_default_run_fn
        run_args = (
            tokenizer,
            "NeelNanda/pile-10k",
            20,
            10,
        )
        weight_config = {
            "*": {
                "data_type": "int",
                "bits": 4,
                "group_size": 32,
                "sym": False,
            }
        }
        quantizer = AutoRoundQuantizer(quant_config=weight_config)

        # quantizer execute
        model = quantizer.prepare(model=model)
        run_fn(model, *run_args)
        q_model = quantizer.convert(model)
        out2 = q_model(**encoded_input)[0]
        assert torch.allclose(out2, out1, atol=0.01), "Accuracy gap atol > 0.01 is unexpected."
