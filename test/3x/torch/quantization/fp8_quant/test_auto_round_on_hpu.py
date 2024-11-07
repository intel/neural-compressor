import copy
import os
import shutil

import pytest
import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config


def set_torch_compile_envs():
    os.environ["PT_HPU_LAZY_MODE"] = "0"
    os.environ["PT_ENABLE_INT64_SUPPORT"] = "1"
    inductor_config.force_disable_caches = True
    dynamo_config.inline_inbuilt_nn_modules = True


set_torch_compile_envs()

import transformers

from neural_compressor.torch.algorithms.weight_only.autoround import get_dataloader
from neural_compressor.torch.quantization import AutoRoundConfig, convert, prepare, quantize
from neural_compressor.torch.utils import logger

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
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

        config = LlamaConfig(num_hidden_layers=2)
        with transformers.modeling_utils.no_init_weights():
            self.tiny_llama_model = AutoModelForCausalLM.from_config(config=config)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.dataloader = get_dataloader(tokenizer, 32, dataset_name="NeelNanda/pile-10k", seed=42, bs=8, nsamples=10)
        self.inp = torch.ones([1, 10], dtype=torch.long)
        self.label = self.tiny_llama_model(self.inp)[0]

    @classmethod
    def teardown_class(self):
        shutil.rmtree("saved_results", ignore_errors=True)

    def setup_method(self, method):
        torch.compiler.reset()
        logger.info(f"Running TestAutoRound test: {method.__name__}")

    @pytest.mark.parametrize("quant_lm_head", [True, False])
    def test_autoround(self, quant_lm_head):
        fp32_model = copy.deepcopy(self.tiny_llama_model)
        quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=10, scale_dtype="fp32")
        if quant_lm_head is False:
            quant_config.set_local("lm_head", AutoRoundConfig(dtype="fp32"))
        logger.info(f"Test AutoRound with config {quant_config}")

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)

        run_fn(model, self.dataloader)
        q_model = convert(model)
        q_model = q_model.cpu()
        self.inp = self.inp
        self.label = self.label
        assert "model.layers.0.self_attn.k_proj" in q_model.autoround_config.keys()
        assert "scale" in q_model.autoround_config["model.layers.0.self_attn.k_proj"].keys()
        assert torch.float32 == q_model.autoround_config["model.layers.0.self_attn.k_proj"]["scale_dtype"]
        assert isinstance(q_model.model.layers[0].self_attn.k_proj, WeightOnlyLinear), "packing model failed."
        if quant_lm_head is True:
            assert isinstance(q_model.lm_head, WeightOnlyLinear), "quantization for lm_head failed."

    def test_int4_dtype(self):
        fp32_model = copy.deepcopy(self.tiny_llama_model)
        quant_config = AutoRoundConfig(dtype="int4", nsamples=32, seqlen=10, iters=10, scale_dtype="fp32")
        logger.info(f"Test AutoRound with config {quant_config}")

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)
        run_fn(model, self.dataloader)
        q_model = convert(model)
        assert "model.layers.0.self_attn.k_proj" in q_model.autoround_config.keys()
        assert "scale" in q_model.autoround_config["model.layers.0.self_attn.k_proj"].keys()
        assert torch.float32 == q_model.autoround_config["model.layers.0.self_attn.k_proj"]["scale_dtype"]
        assert isinstance(q_model.model.layers[0].self_attn.k_proj, WeightOnlyLinear), "packing model failed."
        q_model.cpu()
        out = q_model(self.inp)[0]
        assert out is not None, "Quantization failed!"

    def test_autoround_with_quantize_API(self):
        model = copy.deepcopy(self.tiny_llama_model)

        quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=10, scale_dtype="fp32")
        quant_config.set_local("lm_head", AutoRoundConfig(dtype="fp32"))

        logger.info(f"Test AutoRound with config {quant_config}")

        # quantize API
        q_model = quantize(
            model=model,
            quant_config=quant_config,
            run_fn=run_fn,
            run_args=(self.dataloader,),
        )
        assert isinstance(q_model.model.layers[0].self_attn.k_proj, WeightOnlyLinear), "packing model failed."

    def test_save_and_load(self):
        fp32_model = copy.deepcopy(self.tiny_llama_model)
        quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=10, scale_dtype="fp16")
        # quant_config.set_local("lm_head", AutoRoundConfig(dtype="fp32"))
        logger.info(f"Test AutoRound with config {quant_config}")

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        run_fn(model, self.dataloader)
        q_model = convert(model)

        assert q_model is not None, "Quantization failed!"
        q_model.save("saved_results")
        q_model.cpu()
        inc_out = q_model(self.inp)[0]

        from neural_compressor.torch.algorithms.weight_only.modules import INCWeightOnlyLinear
        from neural_compressor.torch.quantization import load

        # loading compressed model
        loaded_model = load("saved_results", copy.deepcopy(self.tiny_llama_model))
        loaded_out = loaded_model(self.inp)[0]
        assert torch.allclose(inc_out, loaded_out), "Unexpected result. Please double check."
        assert isinstance(
            loaded_model.model.layers[0].self_attn.k_proj, INCWeightOnlyLinear
        ), "loading compressed model failed."
