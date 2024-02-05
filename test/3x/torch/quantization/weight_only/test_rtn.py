import copy

import pytest
import torch
import transformers

from neural_compressor.torch.algorithms.weight_only import WeightOnlyLinear
from neural_compressor.torch.quantization import (
    RTNConfig,
    get_default_double_quant_config,
    get_default_rtn_config,
    quantize,
)


class TestRTNQuant:
    def setup_class(self):
        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
        # record label for comparison
        self.label = self.tiny_gptj(self.example_inputs)[0]
        # test_default_config
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_rtn_config()
        model = quantize(model, quant_config)
        # record q_label for comparison
        self.q_label = model(self.example_inputs)[0]

    def teardown_class(self):
        pass

    @pytest.mark.parametrize(
        "bits, use_sym, group_size, group_dim",
        [
            (8, True, 128, 1),
            (4, True, 128, 1),
            (4, False, 32, 1),
            (4, True, 32, 0),
            (4, False, -1, 1),
            (2, True, 8, 1),
        ],
    )
    def test_int_params(self, bits, use_sym, group_size, group_dim):
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            bits=bits,
            use_sym=use_sym,
            group_size=group_size,
            group_dim=group_dim,
        )
        model = quantize(model, quant_config)
        out = model(self.example_inputs)[0]
        assert (out != self.label).all(), "WOQ output should be different with raw output"
        if (bits, use_sym, group_size, group_dim) == (8, True, 128, 1):
            assert torch.allclose(out, self.label, atol=0.01), "Accuracy gap atol > 0.01 is unexpected."
        if (bits, use_sym, group_size, group_dim) == [(4, True, 128, 0), (4, True, 32, 1)]:
            assert torch.allclose(out, self.label, atol=0.1), "Accuracy gap atol > 0.1 is unexpected."
        if (bits, use_sym, group_size, group_dim) == [(4, False, 32, 0), (4, False, -1, 1), (2, True, 8, 1)]:
            assert torch.allclose(out, self.label, atol=0.5), "Accuracy gap atol > 0.5 is unexpected."

    def test_full_range(self):
        # use_full_range=False, full_range specific to sym
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            use_sym=True,
            use_full_range=False,
        )
        model = quantize(model, quant_config)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.label).amax()
        # use_full_range=True
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            use_sym=True,
            use_full_range=True,
        )
        model = quantize(model, quant_config)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.label).amax()
        # compare atol, this case is an ideal case.
        assert (
            atol_false > atol_true
        ), "use_full_range=True doesn't help accuracy, maybe is reasonable, please double check."

    def test_mse_search(self):
        # use_mse_search=False
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            use_mse_search=False,
        )
        model = quantize(model, quant_config)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.label).amax()
        # use_mse_search=True
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            use_mse_search=True,
        )
        model = quantize(model, quant_config)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.label).amax()
        # compare atol, this case is not an ideal case.
        try:
            assert (
                atol_false > atol_true
            ), "use_mse_search=True doesn't help accuracy, maybe is reasonable, please double check."
        except:
            assert torch.allclose(atol_false, atol_true, atol=0.012), "atol is very close, double checked the logic."

    def test_layer_wise(self):
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            use_layer_wise=True,
        )
        model = quantize(model, quant_config)
        # TODO: (Xin) not implemented

    @pytest.mark.parametrize("dtype", ["int4", "nf4", "fp4"])
    def test_export_compressed_model(self, dtype):
        if dtype == "int4":
            # using optimum format as default
            model = copy.deepcopy(self.tiny_gptj)
            quant_config = RTNConfig(
                dtype=dtype,
                export_compressed_model=True,
            )
            model = quantize(model, quant_config)
            out = model(self.example_inputs)[0]
            assert isinstance(model.lm_head, WeightOnlyLinear), "Exporting compressed model failed."
            atol_true = (out - self.q_label).amax()
            # The small gap is caused by FP16 scale in WeightOnlyLinear.
            assert (
                atol_true < 0.0005
            ), "Exporting compressed model should have the same output as quantized model. Please double check"
        else:
            # optimum_format doesn't suit for symmetric nf4 fp4.
            model = copy.deepcopy(self.tiny_gptj)
            quant_config = RTNConfig(
                dtype=dtype,
                export_compressed_model=False,
            )
            model = quantize(model, quant_config)
            out1 = model(self.example_inputs)[0]
            model = copy.deepcopy(self.tiny_gptj)
            quant_config = RTNConfig(
                dtype=dtype,
                export_compressed_model=True,
            )
            model = quantize(model, quant_config)
            out2 = model(self.example_inputs)[0]
            assert isinstance(model.lm_head, WeightOnlyLinear), "Exporting compressed model failed."
            assert torch.allclose(
                out1, out2
            ), "Exporting compressed model should have the same output as quantized model. Please double check"

    @pytest.mark.parametrize("dtype", ["int4", "nf4", "fp4", "fp4_e2m1_bnb", "fp4_e2m1"])
    def test_dtype_params(self, dtype):
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            dtype=dtype,
        )
        model = quantize(model, quant_config)
        out = model(self.example_inputs)[0]
        assert torch.allclose(out, self.label, atol=0.11), "Accuracy gap atol > 0.11 is unexpected."

    @pytest.mark.parametrize("dtype", ["int4", "nf4"])
    @pytest.mark.parametrize("double_quant_bits", [6])
    @pytest.mark.parametrize("double_quant_group_size", [8, 256])
    # TODO: (Xin) to implement
    # @pytest.mark.parametrize('export_compressed_model', [False, True])
    def test_double_quant_params(self, dtype, double_quant_bits, double_quant_group_size):
        model = copy.deepcopy(self.tiny_gptj)
        # double_quant_use_sym = False
        quant_config = RTNConfig(
            dtype=dtype,
            use_double_quant=True,
            double_quant_bits=double_quant_bits,
            double_quant_use_sym=False,
            double_quant_group_size=double_quant_group_size,
        )
        model = quantize(model, quant_config)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.q_label).amax()
        model = copy.deepcopy(self.tiny_gptj)
        # double_quant_use_sym = True
        quant_config = RTNConfig(
            dtype=dtype,
            use_double_quant=True,
            double_quant_bits=double_quant_bits,
            double_quant_use_sym=True,
            double_quant_group_size=double_quant_group_size,
        )
        model = quantize(model, quant_config)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.q_label).amax()
        # compare atol, this case is an ideal case.
        assert (
            atol_false < atol_true
        ), "asym for double quant should have smaller atol because scales is bigger than zero, please double check."

    def test_double_quant_constants(self):
        model = copy.deepcopy(self.tiny_gptj)
        # the same as get_default_double_quant_config(type="BNB_NF4")
        double_quant_config_dict = get_default_double_quant_config()
        model = quantize(model, double_quant_config_dict)
        out = model(self.example_inputs)[0]
        assert torch.allclose(out, self.label, atol=0.1), "Accuracy gap atol > 0.1 is unexpected."
        # type="BNB_NF4"
        model = copy.deepcopy(self.tiny_gptj)
        double_quant_config_dict = get_default_double_quant_config(type="BNB_NF4")
        model = quantize(model, double_quant_config_dict)
        out1 = model(self.example_inputs)[0]
        atol_BNB = (out1 - self.label).amax()
        assert torch.allclose(out, out1), "Accuracy should be the same, please double check."
        # type="BNB_NF4"
        model = copy.deepcopy(self.tiny_gptj)
        double_quant_config_dict = get_default_double_quant_config(type="GGML_TYPE_Q4_K")
        model = quantize(model, double_quant_config_dict)
        out1 = model(self.example_inputs)[0]
        atol_GGML = (out1 - self.label).amax()
        assert atol_BNB < atol_GGML, "atol_BNB should be smaller than atol_GGML due to its asym double_quant."
