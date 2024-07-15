import copy
import shutil

import pytest
import torch
import transformers

from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedLinear
from neural_compressor.torch.quantization import (
    FP8Config,
    convert,
    finalize_calibration,
    get_default_fp8_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import is_hpex_available


@torch.no_grad()
def calib_func(model):
    example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to("hpu")
    for i in range(2):
        model(example_inputs)


@pytest.mark.skipif(not is_hpex_available(), reason="HPU environment is required!")
class TestFP8StaticQuant:
    def setup_class(self):
        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            device_map="cpu",
        )
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)

    def teardown_class(self):
        shutil.rmtree("test_ouputs", ignore_errors=True)

    def test_one_step_quant(self):
        model = copy.deepcopy(self.tiny_gptj)
        qconfig = FP8Config(fp8_config="E4M3")
        model = prepare(model, qconfig)
        assert isinstance(model.transformer.h[0].attn.k_proj, PatchedLinear), "k_proj is not prepared."
        calib_func(model)
        model = convert(model)
        assert isinstance(model.transformer.h[0].attn.k_proj, PatchedLinear), "k_proj is not quantized."
        assert (
            model.transformer.h[0].attn.k_proj.quant_input.lp_dtype == torch.float8_e4m3fn
        ), "k_proj input dtype is not torch.float8_e4m3fn."

    def test_two_step_quant(self):
        # step 1: measurement
        model = copy.deepcopy(self.tiny_gptj)
        config = FP8Config.from_json_file("test_fp8_jsons/test_measure.json")
        model = prepare(model, config)
        calib_func(model)
        finalize_calibration(model)
        assert isinstance(model.transformer.h[0].attn.k_proj, PatchedLinear), "k_proj is not observed."
        # step 2: quantize based on measurement
        model = copy.deepcopy(self.tiny_gptj)
        config = FP8Config.from_json_file("test_fp8_jsons/test_hw_quant.json")
        model = convert(model, config)
        assert isinstance(model.transformer.h[0].attn.k_proj, PatchedLinear), "k_proj is not quantized."
        assert (
            model.transformer.h[0].attn.k_proj.quant_input.lp_dtype == torch.float8_e4m3fn
        ), "k_proj input dtype is not torch.float8_e4m3fn."
