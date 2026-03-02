import copy
import shutil

import pytest
import torch
import transformers

from neural_compressor.torch.quantization import (
    RTNConfig,
    convert,
    get_default_double_quant_config,
    get_default_rtn_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import accelerator, is_hpu_available

device = accelerator.name()


class ModelConv1d(torch.nn.Module):
    def __init__(self):
        super(ModelConv1d, self).__init__()
        self.fc1 = transformers.Conv1D(64, 32)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def get_woq_linear_num(model, woq_module_type_name):
    woq_linear_num = 0
    for _, module in model.named_modules():
        if module.__class__.__name__ == woq_module_type_name:
            woq_linear_num += 1
    return woq_linear_num


class TestRTNQuant:
    def setup_class(self):
        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            device_map=device,
        )
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to(device)
        # record label for comparison
        self.label = self.tiny_gptj(self.example_inputs)[0]
        # test_default_config
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_rtn_config("Server")
        model = prepare(model, quant_config)
        model = convert(model)
        # record q_label for comparison
        self.q_label = model(self.example_inputs)[0]

    def teardown_class(self):
        shutil.rmtree("saved_results", ignore_errors=True)
        shutil.rmtree("tiny_gptj_int4_rtn", ignore_errors=True)

    # TODO: (4, True, 32, 0), group_dim=0, format not supported
    @pytest.mark.parametrize(
        "bits, use_sym, group_size, group_dim",
        [
            (8, True, -1, 1),
            (4, True, 128, 1),
            (4, False, 32, 1),
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
        model = prepare(model, quant_config)
        model = convert(model)
        out = model(self.example_inputs)[0]
        assert (out != self.label).any(), "WOQ output should be different with raw output"
        if "hpu" in device:
            assert "hpu" in out.device.type, "Neural Compressor should run on HPU when HPEX is available."
        if (bits, use_sym, group_size, group_dim) == (8, True, -1, 1):
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
        model = prepare(model, quant_config)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.label).amax()
        # use_full_range=True
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            use_sym=True,
            use_full_range=True,
        )
        model = prepare(model, quant_config)
        model = convert(model)
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
        model = prepare(model, quant_config)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.label).amax()
        # use_mse_search=True
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            use_mse_search=True,
        )
        model = prepare(model, quant_config)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.label).amax()
        # compare atol, this case is not an ideal case.
        try:
            assert (
                atol_false > atol_true
            ), "use_mse_search=True doesn't help accuracy, maybe is reasonable, please double check."
        except:
            assert torch.allclose(atol_false, atol_true, atol=0.012), "atol is very close, double checked the logic."

    def test_quant_lm_head(self):
        # tie_word_embeddings=false
        gptj_model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            device_map=device,
        )
        lm_head_id = id(gptj_model.lm_head.weight)
        assert id(gptj_model.transformer.wte.weight) != lm_head_id, "The lm_head weight is tied, please check!"
        quant_config = RTNConfig(quant_lm_head=True)
        model = prepare(gptj_model, quant_config)
        model = convert(model)

        # tie_word_embeddings=true
        opt_model = transformers.AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-OPTForCausalLM",
            device_map=device,
        )
        lm_head_id = id(opt_model.lm_head.weight)
        assert (
            id(opt_model.model.decoder.embed_tokens.weight) == lm_head_id
        ), "The lm_head weight is not tied, please check!"
        quant_config = RTNConfig(quant_lm_head=True)
        model = prepare(opt_model, quant_config)
        model = convert(model)
        assert (
            id(model.model.decoder.embed_tokens.weight) == lm_head_id
        ), "The tied lm_head weight is not deep copied, please check!"

    def test_layer_wise(self):
        # use_layer_wise=False
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            use_layer_wise=False,
        )
        model = prepare(model, quant_config)
        model = convert(model)
        out0 = model(self.example_inputs)[0]

        from neural_compressor.torch import load_empty_model

        model = load_empty_model("hf-internal-testing/tiny-random-GPTJForCausalLM")
        quant_config = RTNConfig(
            use_layer_wise=True,
        )
        model = prepare(model, quant_config)
        model = convert(model)
        out1 = model(self.example_inputs)[0]
        assert torch.equal(out1, out0), "use_layer_wise=True output should be same. Please double check."

    @pytest.mark.parametrize(
        "dtype",
        ["int4", "nf4", "fp4", "fp4_e2m1_bnb", "fp4_e2m1", "fp8_e5m2", "fp8_e5m2fnuz", "fp8_e4m3fn", "fp8_e4m3fnuz"],
    )
    def test_dtype_params(self, dtype):
        if dtype in ["fp8_e5m2", "fp8_e5m2fnuz", "fp8_e4m3fn", "fp8_e4m3fnuz"]:
            full_dtype_name = dtype.replace("fp8", "float8")
            if not hasattr(torch, full_dtype_name) or "hpu" in device:
                return  # for low torch version
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(
            dtype=dtype,
        )
        model = prepare(model, quant_config)
        model = convert(model)
        out = model(self.example_inputs)[0]
        out_next = model(self.example_inputs)[0]
        assert torch.allclose(out, self.label, atol=0.11), "Accuracy gap atol > 0.11 is unexpected."
        assert torch.allclose(out, out_next), "output should be same"

    def test_mix_dtype(self):
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig()
        quant_config.set_local(".*mlp.*", RTNConfig(bits=8))
        quant_config.set_local(".*.out_proj", RTNConfig(bits=6))
        quant_config.set_local(".*.k_proj", RTNConfig(dtype="nf4"))
        model = prepare(model, quant_config)
        model = convert(model)
        out = model(self.example_inputs)[0]
        out_next = model(self.example_inputs)[0]
        assert torch.allclose(out, self.label, atol=0.08), "Accuracy gap atol > 0.08 is unexpected."
        assert torch.allclose(out, out_next), "output should be same"

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
        model = prepare(model, quant_config)
        model = convert(model)
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
        model = prepare(model, quant_config)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.q_label).amax()
        # compare atol, this case is an ideal case.
        if not (dtype, double_quant_bits, double_quant_group_size) == ("nf4", 6, 256):
            assert (
                atol_false < atol_true
            ), "asym for double quant should have smaller atol because scales is bigger than zero, please double check."

    def test_double_quant_constants(self):
        model = copy.deepcopy(self.tiny_gptj)
        # the same as get_default_double_quant_config(type="BNB_NF4")
        double_quant_config_dict = get_default_double_quant_config()
        model = prepare(model, double_quant_config_dict)
        model = convert(model)
        out = model(self.example_inputs)[0]
        assert torch.allclose(out, self.label, atol=0.1), "Accuracy gap atol > 0.1 is unexpected."
        # type="BNB_NF4"
        model = copy.deepcopy(self.tiny_gptj)
        double_quant_config_dict = get_default_double_quant_config(type="BNB_NF4")
        model = prepare(model, double_quant_config_dict)
        model = convert(model)
        out1 = model(self.example_inputs)[0]
        assert torch.allclose(out, out1), "Accuracy should be the same, please double check."
        # type="GGML_TYPE_Q4_K"
        model = copy.deepcopy(self.tiny_gptj)
        double_quant_config_dict = get_default_double_quant_config(type="GGML_TYPE_Q4_K")
        model = prepare(model, double_quant_config_dict)
        model = convert(model)
        out2 = model(self.example_inputs)[0]
        assert torch.allclose(out2, self.label, atol=0.1), "Accuracy gap atol > 0.1 is unexpected."

    def test_rtn_with_quantize_API(self):
        quant_config = get_default_rtn_config()

        # prepare + convert API
        model = copy.deepcopy(self.tiny_gptj)
        model = quantize(model, quant_config)
        output_1 = model(self.example_inputs)[0]

        # quantize API
        model = copy.deepcopy(self.tiny_gptj)
        model = prepare(model, quant_config)
        model = convert(model)
        output_2 = model(self.example_inputs)[0]

        # compare the results of calling `convert` + `prepare` and calling `quantize`
        assert torch.all(
            output_1.eq(output_2)
        ), "The results of calling `convert` + `prepare` and calling `quantize` should be equal."

    # TODO: (4, True, 32, 0), group_dim=0, format not supported
    # TODO [SW-216127]: it's not in high priority, so we can implement it later.
    @pytest.mark.skipif(is_hpu_available(), reason="These tests are not supported on HPU for now.")
    @pytest.mark.parametrize(
        "bits, use_sym, group_size, group_dim",
        [
            (8, True, -1, 1),
            (4, True, 128, 1),
            (4, False, 32, 1),
            (4, False, -1, 1),
            (2, True, 8, 1),
        ],
    )
    def test_conv1d(self, bits, use_sym, group_size, group_dim):
        model = ModelConv1d().to(device)
        input = torch.randn(1, 32).to(device)
        quant_config = RTNConfig(
            bits=bits,
            use_sym=use_sym,
            group_size=group_size,
            group_dim=group_dim,
        )
        out1 = model(input)
        model = prepare(model, quant_config)
        model = convert(model)
        out2 = model(input)
        assert (out2 != out1).any(), "WOQ out2put should be different with raw output"
        if (bits, use_sym, group_size, group_dim) == (8, True, -1, 1):
            if "hpu" in device:
                # out2 is float16, no idea.
                assert torch.allclose(out2.float(), out1.float(), atol=0.15), "Accuracy gap atol > 0.15 is unexpected."
            else:
                assert torch.allclose(out2, out1, atol=0.01), "Accuracy gap atol > 0.01 is unexpected."
        if (bits, use_sym, group_size, group_dim) == [(4, True, 128, 0), (4, True, 32, 1)]:
            assert torch.allclose(out2, out1, atol=0.1), "Accuracy gap atol > 0.1 is unexpected."
        if (bits, use_sym, group_size, group_dim) == [(4, False, 32, 0), (4, False, -1, 1), (2, True, 8, 1)]:
            assert torch.allclose(out2, out1, atol=0.5), "Accuracy gap atol > 0.5 is unexpected."

    def test_save_and_load(self):
        from neural_compressor.torch.quantization import load

        fp32_model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_rtn_config()
        q_model = quantize(fp32_model, quant_config=quant_config)
        assert q_model is not None, "Quantization failed!"
        q_model.save("tiny_gptj_int4_rtn", format="huggingface")
        q_model.save("saved_results")

        loaded_model_hf = load("tiny_gptj_int4_rtn", format="huggingface", device=device)
        loaded_model = load("saved_results", copy.deepcopy(self.tiny_gptj), device=device)
        with torch.no_grad():
            inc_out = q_model(self.example_inputs)[0]
            output_hf = loaded_model_hf(self.example_inputs)[0]
            output = loaded_model(self.example_inputs)[0]
        assert torch.allclose(output_hf, output), "Unexpected result. Please double check."
        if "hpu" in device:
            # scale dtype is float16 in saved_results, so we need to set atol=0.002
            assert torch.allclose(inc_out, output, atol=0.002), "Unexpected result. Please double check."
            assert (
                get_woq_linear_num(loaded_model, "HPUWeightOnlyLinear") == 30
            ), "Incorrect number of HPUWeightOnlyLinear modules"
            # test the second load, which reuses the cached quantized_hpu_weight.pt
            loaded_model = load("saved_results", copy.deepcopy(self.tiny_gptj), device="hpu")
            assert (
                get_woq_linear_num(loaded_model, "HPUWeightOnlyLinear") == 30
            ), "Incorrect number of HPUWeightOnlyLinear modules"

            with torch.no_grad():
                output2 = loaded_model(self.example_inputs)[0]
            assert torch.equal(
                output, output2
            ), "The model loaded the second time is different from the model loaded the first time"
        else:
            assert torch.allclose(inc_out, output), "Unexpected result. Please double check."
            assert (
                get_woq_linear_num(loaded_model, "INCWeightOnlyLinear") == 30
            ), "Incorrect number of INCWeightOnlyLinear modules"

    def test_no_transformers(self, monkeypatch):
        def mock_is_transformers_imported():
            return False

        monkeypatch.setattr(
            "neural_compressor.torch.algorithms.weight_only.rtn.is_transformers_imported", mock_is_transformers_imported
        )
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_rtn_config()
        model = prepare(model, quant_config)
        model = convert(model)
        out = model(self.example_inputs)[0]
        assert torch.allclose(out, self.label, atol=1e-1), "Accuracy gap atol > 0.1 is unexpected."

    @pytest.mark.skipif(device == "cpu", reason="no available accelerator")
    def test_auto_host2device(self):
        # if model is on CPU, we move it to device layer-by-layer for acceleration,
        # and then move it back to CPU after quantization.
        model = copy.deepcopy(self.tiny_gptj).to("cpu")
        example_inputs = copy.deepcopy(self.example_inputs).to("cpu")
        quant_config = get_default_rtn_config()
        model = prepare(model, quant_config)
        model = convert(model)
        rtn_label = model(example_inputs)[0]
        rtn_atol = (rtn_label - self.label.to("cpu")).amax()
        assert rtn_atol < 0.08, "RTN should have low atol."
