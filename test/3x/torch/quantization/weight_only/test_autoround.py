import copy
import shutil

import pytest
import torch
import transformers
from packaging.version import Version

from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    convert,
    get_default_AutoRound_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import logger

torch.backends.__allow_nonbracketed_mutation_flag = True
from neural_compressor.torch.algorithms.weight_only.autoround import get_dataloader

try:
    import auto_round
    from auto_round.export.export_to_itrex.model_wrapper import WeightOnlyLinear

    auto_round_installed = True
except ImportError:
    auto_round_installed = False


@torch.no_grad()
def run_fn(model, dataloader):
    for data in dataloader:
        if isinstance(data, tuple) or isinstance(data, list):
            model(*data)
        elif isinstance(data, dict):
            model(**data)
        else:
            model(data)


@pytest.mark.skipif(not auto_round_installed, reason="auto_round module is not installed")
class TestAutoRound:
    @classmethod
    def setup_class(self):
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
        )
        self.inp = torch.ones([1, 10], dtype=torch.long)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM", trust_remote_code=True
        )
        self.dataloader = get_dataloader(tokenizer, 32, dataset_name="NeelNanda/pile-10k", seed=42, bs=8, nsamples=10)
        self.label = self.gptj(self.inp)[0]

    @classmethod
    def teardown_class(self):
        shutil.rmtree("saved_results", ignore_errors=True)

    def setup_method(self, method):
        logger.info(f"Running TestAutoRound test: {method.__name__}")

    @pytest.mark.parametrize("quant_lm_head", [True, False])
    def test_autoround(self, quant_lm_head):
        fp32_model = copy.deepcopy(self.gptj)
        quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=10, scale_dtype="fp32")
        if quant_lm_head is False:
            quant_config.set_local("lm_head", AutoRoundConfig(dtype="fp32"))
        logger.info(f"Test AutoRound with config {quant_config}")

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)

        run_fn(model, self.dataloader)
        q_model = convert(model)
        out = q_model(self.inp)[0]
        assert torch.allclose(out, self.label, atol=1e-1)
        assert "transformer.h.0.attn.k_proj" in q_model.autoround_config.keys()
        assert "scale" in q_model.autoround_config["transformer.h.0.attn.k_proj"].keys()
        assert torch.float32 == q_model.autoround_config["transformer.h.0.attn.k_proj"]["scale_dtype"]
        assert isinstance(q_model.transformer.h[0].attn.k_proj, WeightOnlyLinear), "packing model failed."
        if quant_lm_head is True:
            assert isinstance(q_model.lm_head, WeightOnlyLinear), "quantization for lm_head failed."

    def test_int4_dtype(self):
        fp32_model = copy.deepcopy(self.gptj)
        quant_config = AutoRoundConfig(dtype="int4", nsamples=32, seqlen=10, iters=10, scale_dtype="fp32")
        logger.info(f"Test AutoRound with config {quant_config}")

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)

        run_fn(model, self.dataloader)
        q_model = convert(model)
        out = q_model(self.inp)[0]
        assert torch.allclose(out, self.label, atol=1e-1)
        assert "transformer.h.0.attn.k_proj" in q_model.autoround_config.keys()
        assert "scale" in q_model.autoround_config["transformer.h.0.attn.k_proj"].keys()
        assert torch.float32 == q_model.autoround_config["transformer.h.0.attn.k_proj"]["scale_dtype"]
        assert isinstance(q_model.transformer.h[0].attn.k_proj, WeightOnlyLinear), "packing model failed."

    def test_autoround_with_quantize_API(self):
        gpt_j_model = copy.deepcopy(self.gptj)

        quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=10, scale_dtype="fp32")
        quant_config.set_local("lm_head", AutoRoundConfig(dtype="fp32"))

        logger.info(f"Test AutoRound with config {quant_config}")

        # quantize API
        q_model = quantize(
            model=gpt_j_model,
            quant_config=quant_config,
            run_fn=run_fn,
            run_args=(self.dataloader,),
        )
        out = q_model(self.inp)[0]
        assert torch.allclose(out, self.label, atol=1e-1)
        assert isinstance(q_model.transformer.h[0].attn.k_proj, WeightOnlyLinear), "packing model failed."

    def test_save_and_load(self):
        fp32_model = copy.deepcopy(self.gptj)
        # known issue: scale_dtype="fp32" will cause accuracy gap between quantized model (using auto-round WeightOnlyLinear) and reloaded model (using INCWeightOnlyLinear)
        quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=10, scale_dtype="fp16")
        # quant_config.set_local("lm_head", AutoRoundConfig(dtype="fp32"))
        logger.info(f"Test AutoRound with config {quant_config}")

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        run_fn(model, self.dataloader)
        q_model = convert(model)

        assert q_model is not None, "Quantization failed!"
        q_model.save("saved_results")
        inc_out = q_model(self.inp)[0]

        from neural_compressor.torch.algorithms.weight_only.modules import INCWeightOnlyLinear
        from neural_compressor.torch.quantization import load

        # loading compressed model
        loaded_model = load("saved_results", copy.deepcopy(self.gptj))
        loaded_out = loaded_model(self.inp)[0]
        assert torch.allclose(inc_out, loaded_out), "Unexpected result. Please double check."
        assert isinstance(
            loaded_model.transformer.h[0].attn.k_proj, INCWeightOnlyLinear
        ), "loading compressed model failed."

    def test_conv1d(self):
        input = torch.randn(1, 32)
        from transformers import GPT2Model, GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        out1 = model(**encoded_input)[0]
        quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=10, scale_dtype="fp32")
        model = prepare(model=model, quant_config=quant_config)
        run_fn(model, self.dataloader)
        q_model = convert(model)
        out2 = q_model(**encoded_input)[0]
        assert torch.allclose(out2, out1, atol=0.01), "Accuracy gap atol > 0.01 is unexpected."
        assert isinstance(q_model.h[0].attn.c_attn, WeightOnlyLinear), "loading compressed model failed."

    # def test_autoround_format_export(self):
    #     from neural_compressor.torch.quantization import load
    #     from auto_gptq.nn_modules.qlinear.qlinear_triton import QuantLinear
    #     gpt_j_model = copy.deepcopy(self.gptj)
    #     quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=10, scale_dtype="fp32", export_format="auto_round:gptq")
    #     logger.info(f"Test AutoRound with config {quant_config}")
    #     model = prepare(model=gpt_j_model, quant_config=quant_config)
    #     run_fn(model, self.dataloader)
    #     q_model = convert(model)
    #     out = q_model(self.inp)[0]
    #     assert torch.allclose(out, self.label, atol=1e-1)
    #     assert isinstance(q_model.transformer.h[0].attn.k_proj, QuantLinear), "packing model failed."
    #     q_model.save(output_dir="saved_results_tiny-random-GPTJForCausalLM", format="huggingface")
    #     loaded_model = load("saved_results_tiny-random-GPTJForCausalLM", format="huggingface", trust_remote_code=True)
