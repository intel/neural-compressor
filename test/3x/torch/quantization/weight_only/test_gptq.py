import copy

import pytest
import torch
import transformers

from neural_compressor.torch.quantization import GPTQConfig, get_default_gptq_config, get_default_rtn_config, quantize
from neural_compressor.torch.quantization.modules import WeightOnlyLinear


def run_fn(model):
    # GPTQ uses ValueError to reduce computation when collecting input data of the first block
    # It's special for UTs, no need to add this wrapper in examples.
    with pytest.raises(ValueError):
        model(torch.tensor([[10, 20, 30]], dtype=torch.long))
        model(torch.tensor([[40, 50, 60]], dtype=torch.long))


class TestGPTQQuant:
    def setup_class(self):
        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
        # record label for comparison
        self.label = self.tiny_gptj(self.example_inputs)[0]

    def teardown_class(self):
        pass

    def test_accuracy_improvement(self):
        # test_default_rtn_config
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_rtn_config()
        model = quantize(model, quant_config, run_fn=run_fn)
        rtn_label = model(self.example_inputs)[0]
        rtn_atol = (rtn_label - self.label).amax()
        # test_default_gptq_config
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_gptq_config()
        model = quantize(model, quant_config, run_fn=run_fn)
        gptq_label = model(self.example_inputs)[0]
        gptq_atol = (gptq_label - self.label).amax()
        # 0.05 VS 0.08
        assert gptq_atol < rtn_atol, "GPTQ should have lower atol than RTN, please double check."

    @pytest.mark.parametrize(
        "bits, use_sym, group_size",
        [
            (8, True, 128),
            (4, True, 128),
            (4, False, 32),
            (4, True, 32),
            (4, False, -1),
            (2, True, 8),
        ],
    )
    def test_int_params(self, bits, use_sym, group_size):
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            bits=bits,
            use_sym=use_sym,
            group_size=group_size,
        )
        model = quantize(model, quant_config, run_fn=run_fn)
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
        model = quantize(model, quant_config, run_fn=run_fn)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.label).amax()
        # use_mse_search=True
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            use_mse_search=True,
        )
        model = quantize(model, quant_config, run_fn=run_fn)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.label).amax()
        # compare atol, this case is an ideal case.
        assert (
            atol_false > atol_true
        ), "use_mse_search=True doesn't help accuracy, maybe is reasonable, please double check."

    # def test_layer_wise(self):
    #     model = copy.deepcopy(self.tiny_gptj)
    #     quant_config = GPTQConfig(
    #         use_layer_wise=True,
    #     )
    #     model = quantize(model, quant_config, run_fn=run_fn)
    # TODO: (Xin) not implemented

    @pytest.mark.parametrize("dtype", ["int4", "nf4", "fp4"])
    def test_export_compressed_model(self, dtype):
        # export_compressed_model = False
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            dtype=dtype,
            export_compressed_model=False,
        )
        model = quantize(model, quant_config, run_fn=run_fn)
        out1 = model(self.example_inputs)[0]
        # export_compressed_model = True
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            dtype=dtype,
            export_compressed_model=True,
        )
        model = quantize(model, quant_config, run_fn=run_fn)
        out2 = model(self.example_inputs)[0]
        assert isinstance(model.transformer.h[0].attn.k_proj, WeightOnlyLinear), "Exporting compressed model failed."

        # The small gap is caused by FP16 scale in WeightOnlyLinear.
        if dtype == "int4":
            atol_true = (out1 - out2).amax()
            assert (
                atol_true < 0.008
            ), "Exporting compressed model should have the same output as quantized model. Please double check"
        else:
            assert torch.allclose(
                out1, out2
            ), "Exporting compressed model should have the same output as quantized model. Please double check."

    @pytest.mark.parametrize("dtype", ["int4", "nf4", "fp4", "fp4_e2m1_bnb", "fp4_e2m1"])
    def test_dtype_params(self, dtype):
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            dtype=dtype,
        )
        model = quantize(model, quant_config, run_fn=run_fn)
        out = model(self.example_inputs)[0]
        atol = (out - self.label).amax()
        assert atol < 0.12, "Accuracy gap atol > 0.12 is unexpected. Please double check."

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
        model = quantize(model, quant_config, run_fn=run_fn)
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
        model = quantize(model, quant_config, run_fn=run_fn)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.label).amax()
        # compare atol, this case is not an ideal case.
        try:
            assert (
                atol_false < atol_true
            ), "asym for double quant should have smaller atol because scales is bigger than zero, please double check."
        except:
            assert torch.allclose(atol_false, atol_true, atol=0.008), "atol is very close, double checked the logic."
