import copy
import os
import shutil

import pytest
import torch
import transformers

from neural_compressor.torch.quantization import (
    GPTQConfig,
    convert,
    get_default_gptq_config,
    get_default_rtn_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import accelerator, is_hpu_available

device = accelerator.name()


def run_fn(model):
    model(torch.tensor([[10, 20, 30]], dtype=torch.long).to(device))


def get_woq_linear_num(model, woq_module_type_name):
    woq_linear_num = 0
    for _, module in model.named_modules():
        if module.__class__.__name__ == woq_module_type_name:
            woq_linear_num += 1
    return woq_linear_num


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
        shutil.rmtree("tiny_gptj_int4_gptq", ignore_errors=True)

    @pytest.mark.skipif(device == "cpu", reason="no available accelerator")
    def test_auto_host2device(self):
        # if model is on CPU, we move it to device layer-by-layer for acceleration,
        # and then move it back to CPU after quantization.
        model = copy.deepcopy(self.tiny_gptj).to("cpu")
        example_inputs = copy.deepcopy(self.example_inputs).to("cpu")
        quant_config = get_default_gptq_config()
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        gptq_label = model(example_inputs)[0]
        gptq_atol = (gptq_label - self.label.to("cpu")).amax()
        assert gptq_atol < 0.06, "GPTQ should have low atol."

    def test_accuracy_improvement(self):
        # test_default_rtn_config
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_rtn_config()
        model = prepare(model, quant_config)
        run_fn(model)
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

    @pytest.mark.skipif(not is_hpu_available(), reason="These tests are not supported on HPU for now.")
    def test_block_wise(self):
        from neural_compressor.torch import load_empty_model
        from neural_compressor.torch.algorithms.layer_wise import load_first_layer_only
        from neural_compressor.torch.algorithms.layer_wise.utils import LWQ_WORKSPACE
        from neural_compressor.torch.quantization import load
        from neural_compressor.torch.utils import get_used_cpu_mem_MB, get_used_hpu_mem_MB

        model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig()
        cpu_mem0 = get_used_cpu_mem_MB()
        hpu_mem0 = get_used_hpu_mem_MB()

        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        cpu_mem_diff = get_used_cpu_mem_MB() - cpu_mem0
        hpu_mem_diff = get_used_hpu_mem_MB() - hpu_mem0
        q_label = model(self.example_inputs)[0]

        model = load_empty_model(model_name)
        load_first_layer_only(model, model_name)

        quant_config = GPTQConfig(
            use_block_wise=True,
            model_path=model_name,
        )
        cpu_mem0 = get_used_cpu_mem_MB()
        hpu_mem0 = get_used_hpu_mem_MB()
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        cpu_mem_block_diff = get_used_cpu_mem_MB() - cpu_mem0
        hpu_mem_block_diff = get_used_hpu_mem_MB() - hpu_mem0

        kwargs = {"blockwise": True, "blockwise_load_folder": None}
        model.save(LWQ_WORKSPACE + "/checkpoint/", **kwargs)

        kwargs = {"sharded_checkpoints": True}

        loaded_model = load(LWQ_WORKSPACE + "/checkpoint/", copy.deepcopy(self.tiny_gptj), **kwargs).to(device)

        out = loaded_model(self.example_inputs)[0]

        # remove lwq tmp directory
        shutil.rmtree(LWQ_WORKSPACE, ignore_errors=True)

        assert torch.allclose(out, q_label, atol=0.05), "block-wise and none-blockwise shouold be almost identical."

        assert cpu_mem_diff > cpu_mem_block_diff, "block-wise should reduce memory."
        assert hpu_mem_diff > hpu_mem_block_diff, "block-wise should reduce memory."

    @pytest.mark.parametrize("quant_lm_head", [False, True])
    def test_layer_wise(self, quant_lm_head):
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(quant_lm_head=quant_lm_head)
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        q_label = model(self.example_inputs)[0]

        from neural_compressor.torch import load_empty_model

        model = load_empty_model("hf-internal-testing/tiny-random-GPTJForCausalLM")

        quant_config = GPTQConfig(
            use_layer_wise=True,
            quant_lm_head=quant_lm_head,
            model_path="hf-internal-testing/tiny-random-GPTJForCausalLM",
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]

        # remove lwq tmp directory
        from neural_compressor.torch.algorithms.layer_wise.utils import LWQ_WORKSPACE

        shutil.rmtree(LWQ_WORKSPACE, ignore_errors=True)
        assert torch.equal(
            out, q_label
        ), f"use_layer_wise=True and quant_lm_head={quant_lm_head} output should be same. Please double check."

    def test_true_sequential(self):
        # true_sequential=False
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            true_sequential=False,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.label).amax()
        # true_sequential=True
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            true_sequential=True,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.label).amax()
        # compare atol, this case is an ideal case.
        assert (
            atol_false < atol_true
        ), "true_sequential=True doesn't help accuracy, maybe is reasonable, please double check."

    # TODO [SW-216127]: it's not in high priority, so we can implement it later.
    @pytest.mark.skipif(is_hpu_available(), reason="These tests are not supported on HPU for now.")
    def test_quant_lm_head(self):
        # quant_lm_head=False
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            quant_lm_head=False,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_false = (out - self.label).amax()
        # quant_lm_head=True
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = GPTQConfig(
            quant_lm_head=True,
        )
        model = prepare(model, quant_config)
        run_fn(model)
        model = convert(model)
        out = model(self.example_inputs)[0]
        atol_true = (out - self.label).amax()
        # compare atol, this case is an ideal case.
        assert (
            atol_false < atol_true
        ), "quant_lm_head=True doesn't help accuracy, maybe is reasonable, please double check."
        assert get_woq_linear_num(model, "INCWeightOnlyLinear") == 31, "Incorrect number of INCWeightOnlyLinear modules"

    @pytest.mark.parametrize("dtype", ["nf4", "int4"])
    @pytest.mark.parametrize("double_quant_bits", [6])
    @pytest.mark.parametrize("double_quant_group_size", [8, 256])
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

    # TODO [SW-216127]: it's not in high priority, so we can implement it later.
    @pytest.mark.skipif(is_hpu_available(), reason="These tests are not supported on HPU for now.")
    def test_conv1d(self):
        from transformers import GPT2Model, GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = GPT2Model.from_pretrained("sshleifer/tiny-gpt2").to(device)
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt").to(device)

        def run_fn_conv1d(model):
            model(**encoded_input)

        quant_config = get_default_gptq_config()
        out1 = model(**encoded_input)[0]
        model = prepare(model, quant_config)
        run_fn_conv1d(model)
        q_model = convert(model)
        out2 = q_model(**encoded_input)[0]
        assert torch.allclose(out2, out1, atol=0.01), "Accuracy gap atol > 0.01 is unexpected."

    def test_save_and_load(self):
        from neural_compressor.torch.quantization import load

        fp32_model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_gptq_config()
        prepared_model = prepare(fp32_model, quant_config)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"
        q_model.save("tiny_gptj_int4_gptq", format="huggingface")
        q_model.save("saved_results")

        loaded_model_hf = load("tiny_gptj_int4_gptq", format="huggingface", device=device)
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
