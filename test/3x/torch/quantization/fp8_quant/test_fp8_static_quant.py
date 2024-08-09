import copy
import shutil

import pytest
import torch
import torchvision
import transformers

from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedConv2d, PatchedLinear
from neural_compressor.torch.quantization import (
    FP8Config,
    convert,
    finalize_calibration,
    get_default_fp8_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import is_hpex_available


def change_to_cur_file_dir():
    import os

    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    os.chdir(current_directory)


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
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to("hpu")
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.cv_dummy_inputs = torch.randn([1, 3, 224, 224]).to("hpu")

    def teardown_class(self):
        shutil.rmtree("test_ouputs", ignore_errors=True)

    def test_one_step_quant_nlp(self):
        model = copy.deepcopy(self.tiny_gptj)
        model.to("hpu")
        fp32_out = model(self.example_inputs)[0]
        qconfig = FP8Config(fp8_config="E4M3")
        model = prepare(model, qconfig)
        assert isinstance(model.transformer.h[0].attn.k_proj, PatchedLinear), "k_proj is not prepared."
        calib_func(model)
        model = convert(model)
        fp8_out = model(self.example_inputs)[0]
        assert isinstance(model.transformer.h[0].attn.k_proj, PatchedLinear), "k_proj is not quantized."
        assert (
            model.transformer.h[0].attn.k_proj.quant_input.lp_dtype == torch.float8_e4m3fn
        ), "k_proj input dtype is not torch.float8_e4m3fn."
        assert (fp32_out != fp8_out).any(), "FP32 output should be different with FP8 output"
        print((fp32_out - fp8_out).abs().max())
        assert torch.allclose(fp32_out, fp8_out, atol=0.04), "Accuracy gap atol > 0.04 is unexpected."

    # @pytest.mark.skipif(not is_hpex_available(), reason="HPU environment is required!")
    def test_one_step_quant_cv(self):
        model = copy.deepcopy(self.resnet18)
        model.to("hpu")
        fp32_out = model(self.cv_dummy_inputs)
        # model.to('cpu')
        qconfig = FP8Config(fp8_config="E4M3")
        model = prepare(model, qconfig)
        assert model.fc.weight.device.type == "hpu", "model is not mapped to HPU."
        assert isinstance(model.fc, PatchedLinear) and isinstance(model.conv1, PatchedConv2d), "model is not prepared."
        # calibration
        model(self.cv_dummy_inputs)
        model = convert(model)
        fp8_out = model(self.cv_dummy_inputs)
        assert (
            isinstance(model.fc, PatchedLinear)
            and isinstance(model.conv1, PatchedConv2d)
            and model.fc.quant_input.lp_dtype == torch.float8_e4m3fn
            and model.conv1.quant_input.lp_dtype == torch.float8_e4m3fn
        ), "model is not quantized to torch.float8_e4m3fn."
        assert (fp32_out != fp8_out).any(), "FP32 output should be different with FP8 output"

    def test_two_step_quant_nlp(self):
        change_to_cur_file_dir()
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

    def test_two_step_quant_cv(self):
        change_to_cur_file_dir()
        # step 1: measurement
        model = copy.deepcopy(self.resnet18)
        config = FP8Config.from_json_file("test_fp8_jsons/test_measure.json")
        model = prepare(model, config)
        fp32_out = model(self.cv_dummy_inputs)
        finalize_calibration(model)
        assert isinstance(model.fc, PatchedLinear) and isinstance(model.conv1, PatchedConv2d), "model is not prepared."
        # step 2: quantize based on measurement
        model = copy.deepcopy(self.resnet18)
        config = FP8Config.from_json_file("test_fp8_jsons/test_hw_quant.json")
        model = convert(model, config)
        fp8_out = model(self.cv_dummy_inputs)
        assert (
            isinstance(model.fc, PatchedLinear)
            and isinstance(model.conv1, PatchedConv2d)
            and model.fc.quant_input.lp_dtype == torch.float8_e4m3fn
            and model.conv1.quant_input.lp_dtype == torch.float8_e4m3fn
        ), "model is not quantized to torch.float8_e4m3fn."
        assert (fp32_out != fp8_out).any(), "FP32 output should be different with FP8 output"
