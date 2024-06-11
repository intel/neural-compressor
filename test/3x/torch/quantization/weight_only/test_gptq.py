import copy
import shutil

import pytest
import torch
import transformers

from neural_compressor.torch.algorithms.weight_only.modules import WeightOnlyLinear
from neural_compressor.torch.quantization import (
    GPTQConfig,
    convert,
    get_default_gptq_config,
    get_default_rtn_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import accelerator

device = accelerator.current_device_name()


def run_fn_for_rtn(model):
    model(torch.tensor([[10, 20, 30]], dtype=torch.long).to(device))
    model(torch.tensor([[40, 50, 60]], dtype=torch.long).to(device))


def run_fn(model):
    # GPTQ uses ValueError to reduce computation when collecting input data of the first block
    # It's special for UTs, no need to add this wrapper in examples.
    with pytest.raises(ValueError):
        model(torch.tensor([[10, 20, 30]], dtype=torch.long).to(device))
        model(torch.tensor([[40, 50, 60]], dtype=torch.long).to(device))


class TestGPTQQuant:
    def setup_class(self):
        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            device_map=device,
        )
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to(device)
        # record label for comparison
        self.label = self.tiny_gptj(self.example_inputs)[0]

    def teardown_class(self):
        shutil.rmtree("saved_results", ignore_errors=True)

    def test_accuracy_improvement(self):
        # test_default_rtn_config
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_rtn_config()
        model = prepare(model, quant_config)
        run_fn_for_rtn(model)
        model = convert(model)
        rtn_label = model(self.example_inputs)[0]
        rtn_atol = (rtn_label - self.label).amax()
        # test_default_gptq_config
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_gptq_config()
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        gptq_label = model(self.example_inputs)[0]
        gptq_atol = (gptq_label - self.label).amax()
        # 0.05 VS 0.08
        assert gptq_atol < rtn_atol, "GPTQ should have lower atol than RTN, please double check."

    def test_gptq_with_quantize_API(self):
        # test_default_gptq_config
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_gptq_config()

        # prepare + convert API
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        gptq_label = model(self.example_inputs)[0]
        gptq_atol_1 = (gptq_label - self.label).amax()

        # quantize API
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_gptq_config()
        model = quantize(model, quant_config, run_fn=run_fn)
        gptq_label = model(self.example_inputs)[0]
        gptq_atol_2 = (gptq_label - self.label).amax()

        # compare the results of calling `convert` + `prepare` and calling `quantize`
        assert (
            gptq_atol_1 == gptq_atol_2
        ), "The results of calling `convert` + `prepare` and calling `quantize` should be equal."

    @pytest.mark.parametrize(
        "bits, use_sym, group_size, act_order",
        [
            (8, True, 128, False),
            (4, True, 128, False),
            (4, False, 32, False),
            (4, True, 32, True),
            (4, False, -1, True),
            (2, True, 8, True),
        ],
    )
    def test_int_params(self, bits, use_sym, group_size, act_order):
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            bits=bits,
            use_sym=use_sym,
            group_size=group_size,
            act_order=act_order,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        assert (out != self.label).all(), "WOQ output should be different with raw output"
        if (bits, use_sym, group_size) == (8, True, 128):
            assert torch.allclose(out, self.label, atol=0.005), "Accuracy gap atol > 0.005 is unexpected."
        if (bits, use_sym, group_size) == [(4, True, 128), (4, True, 32), (4, False, 32), (4, False, -1)]:
            assert torch.allclose(out, self.label, atol=0.08), "Accuracy gap atol > 0.08 is unexpected."
        if (bits, use_sym, group_size) == [(2, True, 8)]:
            assert torch.allclose(out, self.label, atol=0.25), "Accuracy gap atol > 0.25 is unexpected."

    def test_mse_search(self):
        # use_mse_search=False
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            use_mse_search=False,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.label).amax()
        # use_mse_search=True
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            use_mse_search=True,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.label).amax()
        # compare atol, this case is an ideal case.
        assert (
            atol_false > atol_true
        ), "use_mse_search=True doesn't help accuracy, maybe is reasonable, please double check."

    def test_act_order(self):
        # act_order=False
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            act_order=False,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.label).amax()
        # act_order=True
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            act_order=True,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.label).amax()
        # compare atol, this case is an ideal case.
        assert atol_false > atol_true, "act_order=True doesn't help accuracy, maybe is reasonable, please double check."

    # def test_layer_wise(self):
    #     model = copy.deepcopy(self.tiny_gptj)
    #     quant_config = GPTQConfig(
    #         use_layer_wise=True,
    #     )
    #     model = quantize(model, quant_config, run_fn=run_fn)
    # TODO: (Xin) not implemented

    @pytest.mark.parametrize("dtype", ["nf4", "int4"])
    @pytest.mark.parametrize("double_quant_bits", [6])
    @pytest.mark.parametrize("double_quant_group_size", [8, 256])
    # TODO: (Xin) to implement
    # @pytest.mark.parametrize('export_compressed_model', [False, True])
    def test_double_quant_params(self, dtype, double_quant_bits, double_quant_group_size):
        model = copy.deepcopy(self.tiny_gptj)
        # double_quant_use_sym = False
        quant_config = GPTQConfig(
            dtype=dtype,
            use_double_quant=True,
            double_quant_bits=double_quant_bits,
            double_quant_use_sym=False,
            double_quant_group_size=double_quant_group_size,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.label).amax()
        model = copy.deepcopy(self.tiny_gptj)
        # double_quant_use_sym = True
        quant_config = GPTQConfig(
            dtype=dtype,
            use_double_quant=True,
            double_quant_bits=double_quant_bits,
            double_quant_use_sym=True,
            double_quant_group_size=double_quant_group_size,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.label).amax()
        # compare atol, this case is not an ideal case.
        try:
            assert (
                atol_false < atol_true
            ), "asym for double quant should have smaller atol because scales is bigger than zero, please double check."
        except:
            assert torch.allclose(atol_false, atol_true, atol=0.008), "atol is very close, double checked the logic."

    def test_conv1d(self):
        from transformers import GPT2Model, GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")

        def run_fn_conv1d(model):
            with pytest.raises(ValueError):
                for i in range(2):
                    model(**encoded_input)

        quant_config = get_default_gptq_config()
        out1 = model(**encoded_input)[0]
        model = prepare(model, quant_config)
        run_fn_conv1d(model)
        q_model = convert(model)
        out2 = q_model(**encoded_input)[0]
        assert torch.allclose(out2, out1, atol=0.01), "Accuracy gap atol > 0.01 is unexpected."

    def test_save_and_load(self):
        fp32_model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_gptq_config()
        prepared_model = prepare(fp32_model, quant_config)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"
        q_model.save("saved_results")
        inc_out = q_model(self.example_inputs)[0]

        from neural_compressor.torch.quantization import load

        # loading compressed model
        loaded_model = load("saved_results", copy.deepcopy(self.tiny_gptj))
        loaded_out = loaded_model(self.example_inputs)[0]
        assert torch.allclose(inc_out, loaded_out), "Unexpected result. Please double check."
        assert isinstance(
            loaded_model.transformer.h[0].attn.k_proj, WeightOnlyLinear
        ), "loading compressed model failed."
