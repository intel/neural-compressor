import copy
import os
import time
from copy import deepcopy

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM

from neural_compressor.common import options
from neural_compressor.common.utils import logger
from neural_compressor.torch.algorithms.weight_only.hqq.config import HQQModuleConfig, QTensorConfig, hqq_global_option
from neural_compressor.torch.algorithms.weight_only.hqq.core import HQQLinear
from neural_compressor.torch.quantization import HQQConfig, convert, get_default_hqq_config, prepare, quantize
from neural_compressor.torch.utils import accelerator

device = accelerator.current_device_name()


def _common_hqq_test(
    nbits=4, group_size=64, quant_zero=True, quant_scale=False, scale_quant_group_size=128, device=None
):
    # Parse config
    weight_qconfig = QTensorConfig(
        nbits=nbits, channel_wise=True, group_size=group_size, optimize=True, round_zero=True if nbits == 4 else False
    )
    zero_qconfig = None
    if quant_zero:
        zero_qconfig = QTensorConfig(nbits=8, channel_wise=False, group_size=None, optimize=False)
    scale_qconfig = None
    if quant_scale:
        scale_qconfig = QTensorConfig(nbits=8, channel_wise=True, group_size=scale_quant_group_size, optimize=False)
    hqq_quant_config = HQQModuleConfig(weight=weight_qconfig, scale=scale_qconfig, zero=zero_qconfig)

    # Create HQQ Linear
    bs = 4
    in_features = 64
    out_features = 128
    float_linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
    if hqq_global_option.use_half:
        logger.info(f"hqq_global_option use half: {hqq_global_option.use_half}")
        float_linear = float_linear.half()
    float_linear.to(device)
    float_linear_copy = deepcopy(float_linear)
    hqq_linear = HQQLinear.from_float(float_linear_copy, quant_config=hqq_quant_config)

    # Forward
    input = torch.randn(bs, in_features, device=device)
    if hqq_global_option.use_half:
        input = input.half()
    float_output = float_linear(input)
    input_for_hqq = deepcopy(input)
    hqq_output = hqq_linear(input_for_hqq)
    hqq_output_2 = hqq_linear(input_for_hqq)
    torch.allclose(float_output, hqq_output, atol=0.5)
    torch.allclose(hqq_output, hqq_output_2)
    del float_linear, hqq_linear
    del float_output, hqq_output, hqq_output_2


class TestHQQ:

    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

    @pytest.fixture
    def force_use_cpu(self, monkeypatch):
        # Force use CPU
        monkeypatch.setenv("INC_TARGET_DEVICE", "cpu")

    @pytest.fixture
    def force_not_half(self, monkeypatch):
        monkeypatch.setattr(hqq_global_option, "use_half", False)

    def test_hqq_quant(self, force_use_cpu, force_not_half):

        hqq_global_option.use_half = False
        fp32_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-OPTForCausalLM")
        example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long, device="cpu")
        # test_default_config
        quant_config = get_default_hqq_config()

        # prepare + convert API
        model = prepare(deepcopy(fp32_model), quant_config)
        model = convert(model)
        q_label_1 = model(example_inputs)[0]

        # quantize API
        model = quantize(deepcopy(fp32_model), quant_config)
        q_label_2 = model(example_inputs)[0]

        # compare the results of calling `convert` + `prepare` and calling `quantize`
        assert torch.all(
            q_label_1.eq(q_label_2)
        ), "The results of calling `convert` + `prepare` and calling `quantize` should be equal."

    def test_hqq_load_save(self, force_use_cpu, force_not_half):

        hqq_global_option.use_half = False
        fp32_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-OPTForCausalLM")
        example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long, device="cpu")
        # test_default_config
        quant_config = get_default_hqq_config()

        # prepare + convert API
        model = prepare(deepcopy(fp32_model), quant_config)
        qmodel = convert(model)
        qmodel_out_ref = model(example_inputs)[0]
        save_path = options.workspace + f"/_hqq_model_{time.time()}.pth"
        qmodel.save(save_path)
        from neural_compressor.torch.quantization import load

        # loading compressed model
        loaded_model = load(save_path, copy.deepcopy(fp32_model))
        loaded_model_out = loaded_model(example_inputs)[0]
        assert torch.allclose(qmodel_out_ref, loaded_model_out), "Unexpected result. Please double check."

    def test_hqq_fallback(self, force_use_cpu, force_not_half):

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(128, 1024)
                self.fc2 = torch.nn.Linear(1024, 512)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        quant_config = HQQConfig().set_local("fc1", HQQConfig(dtype="fp32"))
        qmodel = convert(prepare(model=ToyModel(), quant_config=quant_config))
        assert type(qmodel.fc1).__name__ == torch.nn.Linear.__name__, f"Expect fallback fc1, but get {type(qmodel.fc1)}"
        assert type(qmodel.fc2).__name__ != torch.nn.Linear.__name__, f"Expect quantize fc2, but get {type(qmodel.fc2)}"

    def test_quant_lm_head(self, force_use_cpu, force_not_half):
        # tie_word_embeddings=false
        gptj_model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            device_map=device,
        )
        lm_head_id = id(gptj_model.lm_head.weight)
        assert id(gptj_model.transformer.wte.weight) != lm_head_id, "The lm_head weight is tied, please check!"
        quant_config = HQQConfig(quant_lm_head=True)
        model = prepare(gptj_model, quant_config)
        model = convert(model)

        # tie_word_embeddings=true
        opt_model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",  # group_size should be divisible by tensor.numel(). Dummy model cannot work.
            device_map=device,
        )
        lm_head_id = id(opt_model.lm_head.weight)
        assert (
            id(opt_model.model.decoder.embed_tokens.weight) == lm_head_id
        ), "The lm_head weight is not tied, please check!"
        quant_config = HQQConfig(quant_lm_head=True)
        model = prepare(opt_model, quant_config)
        model = convert(model)
        assert (
            id(model.model.decoder.embed_tokens.weight) == lm_head_id
        ), "The tied lm_head weight is not deep copied, please check!"

    @pytest.mark.parametrize("device_name", ["cuda", "cpu"])
    @pytest.mark.parametrize(
        "nbits, group_size, quant_zero, quant_scale, scale_quant_group_size",
        [
            (4, 64, True, False, 128),
            (4, 64, False, False, 128),
            (4, 64, True, True, 128),
            (4, 64, False, True, 128),
            (8, 64, True, False, 128),
            (8, 64, False, False, 128),
            (8, 64, True, True, 128),
            (8, 64, False, True, 128),
            (4, 64, True, False, 64),
            (4, 64, False, False, 64),
            (4, 64, True, True, 64),
            (4, 64, False, True, 64),
            (4, -1, False, True, 64),
        ],
    )
    def test_hqq_module(
        self,
        nbits,
        group_size,
        quant_zero,
        quant_scale,
        scale_quant_group_size,
        device_name,
    ):
        if device_name == "cuda" and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test because cuda is not available")
        if device_name == "cpu":
            os.environ["INC_TARGET_DEVICE"] = "cpu"
            hqq_global_option.use_half = False

        _common_hqq_test(
            nbits=nbits,
            group_size=group_size,
            quant_zero=quant_zero,
            quant_scale=quant_scale,
            scale_quant_group_size=scale_quant_group_size,
            device=torch.device(device_name),
        )

    @pytest.mark.parametrize(
        "nbits, group_size, quant_zero, quant_scale, scale_quant_group_size",
        [
            (4, 64, True, False, 128),
            (4, 64, False, False, 128),
            (4, 64, True, True, 128),
            (4, 64, False, True, 128),
            (8, 64, True, False, 128),
        ],
    )
    def test_hqq_linear_save_and_load(
        self,
        nbits,
        group_size,
        quant_zero,
        quant_scale,
        scale_quant_group_size,
    ):
        hqq_global_option.use_half = False
        # Parse config
        weight_qconfig = QTensorConfig(
            nbits=nbits,
            channel_wise=True,
            group_size=group_size,
            optimize=True,
            round_zero=True if nbits == 4 else False,
        )
        zero_qconfig = None
        if quant_zero:
            zero_qconfig = QTensorConfig(nbits=8, channel_wise=False, group_size=None, optimize=False)
        scale_qconfig = None
        if quant_scale:
            scale_qconfig = QTensorConfig(nbits=8, channel_wise=True, group_size=scale_quant_group_size, optimize=False)
        hqq_quant_config = HQQModuleConfig(weight=weight_qconfig, scale=scale_qconfig, zero=zero_qconfig)
        # Create HQQ Linear
        bs = 4
        in_features = 64
        out_features = 128
        float_linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
        float_linear.to(device)
        float_linear_copy = deepcopy(float_linear)
        input = torch.randn(bs, in_features, device=device)
        hqq_linear = HQQLinear.from_float(float_linear_copy, quant_config=hqq_quant_config)
        out_ref = hqq_linear(input)
        state_dict = hqq_linear.state_dict()
        hqq_module_path = options.workspace + f"/_hqq_linear_{time.time()}.pth"
        torch.save(state_dict, hqq_module_path)
        reload_state_dict = torch.load(hqq_module_path)
        new_float = torch.nn.Linear(in_features=in_features, out_features=out_features)
        new_hqq_linear = HQQLinear.from_float(new_float, quant_config=hqq_quant_config)
        new_hqq_linear.load_state_dict(reload_state_dict)
        out = new_hqq_linear(input)
        assert torch.equal(out_ref, out), f"out_ref: {out_ref}, out: {out}"
