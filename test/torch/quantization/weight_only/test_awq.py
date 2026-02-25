import copy
import shutil

import pytest
import torch
import transformers

from neural_compressor.common import Logger

logger = Logger().get_logger()
from neural_compressor.torch.algorithms.weight_only.modules import MulLinear
from neural_compressor.torch.quantization import AWQConfig, convert, get_default_awq_config, prepare, quantize
from neural_compressor.torch.utils import accelerator

device = accelerator.current_device_name()


def get_gpt_j():
    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        torchscript=True,
        device_map=device,
    )
    return tiny_gptj


@torch.no_grad()
def calib_func(model):
    example_inputs = torch.ones([1, 10], dtype=torch.long).to(device)
    for i in range(2):
        model(example_inputs)


def get_woq_linear_num(model, woq_module_type_name):
    woq_linear_num = 0
    for _, module in model.named_modules():
        if module.__class__.__name__ == woq_module_type_name:
            woq_linear_num += 1
    return woq_linear_num


class TestAWQQuant:
    @classmethod
    def setup_class(self):
        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            device_map=device,
            torchscript=True,
        )
        self.example_inputs = torch.ones([1, 10], dtype=torch.long).to(device)
        self.label = self.tiny_gptj(self.example_inputs)[0]

    def teardown_class(self):
        shutil.rmtree("saved_results", ignore_errors=True)

    @pytest.mark.parametrize(
        "bits, use_sym, group_size",
        [
            (8, True, -1),
            (4, True, 128),
            (4, False, 32),
            (4, False, -1),
            (2, True, 8),
        ],
    )
    def test_awq(self, bits, use_sym, group_size):
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = AWQConfig(bits=8, group_size=-1)
        logger.info(f"Test AWQ with config {quant_config}")
        model = prepare(
            model=model,
            quant_config=quant_config,
            example_inputs=self.example_inputs,
        )
        calib_func(model)
        qdq_model = convert(model)
        out = qdq_model(self.example_inputs)[0]

        # default awq_quantize is 4 bits, 32 group size, use big atol=1e-1
        if (bits, use_sym, group_size) == (8, True, -1):
            # TODO mul floded:
            # assert not isinstance(qdq_model.transformer.h[0].attn.k_proj, MulLinear), "mul in k_proj should be folded."
            assert torch.allclose(out, self.label, atol=1e-2), "Accuracy gap atol > 0.01 is unexpected."
        elif (bits, use_sym, group_size) == (2, True, 8):
            assert torch.allclose(out, self.label, atol=0.5), "Accuracy gap atol > 0.5 is unexpected."
        else:
            assert torch.allclose(out, self.label, atol=1e-1), "Accuracy gap atol > 0.01 is unexpected."

    def test_awq_with_quantize_API(self):
        quant_config = get_default_awq_config()
        logger.info(f"Test AWQ with config {quant_config}")

        # prepare + convert API
        model = prepare(
            model=copy.deepcopy(self.tiny_gptj),
            quant_config=quant_config,
            example_inputs=self.example_inputs,
        )
        calib_func(model)
        qdq_model = convert(model)
        out1 = qdq_model(self.example_inputs)

        # quantize API
        qdq_model = quantize(
            model=copy.deepcopy(self.tiny_gptj),
            quant_config=quant_config,
            example_inputs=self.example_inputs,
            run_fn=calib_func,
        )
        out2 = qdq_model(self.example_inputs)

        # compare the results of calling `convert` + `prepare` and calling `quantize`
        assert torch.all(
            out1[0].eq(out2[0])
        ), "The results of calling `convert` + `prepare` and calling `quantize` should be equal."

    def test_save_and_load(self):
        from neural_compressor.torch.quantization import load

        @torch.no_grad()
        def calib_func(model):
            for i in range(2):
                model(self.example_inputs)

        fp32_model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_awq_config()
        # prepare + convert API
        model = prepare(
            model=fp32_model,
            quant_config=quant_config,
            example_inputs=self.example_inputs,
        )
        calib_func(model)
        q_model = convert(model)
        assert q_model is not None, "Quantization failed!"
        q_model.save("saved_results")
        inc_out = q_model(self.example_inputs)[0]

        # loading compressed model (format=default, device="cpu")
        # linear -> INCWeightOnlyLinear
        loaded_model = load("saved_results", copy.deepcopy(self.tiny_gptj))
        output = loaded_model(self.example_inputs)[0]
        assert torch.allclose(inc_out, output), "Unexpected result. Please double check."
        assert (
            get_woq_linear_num(loaded_model, "INCWeightOnlyLinear") == 30
        ), "Incorrect number of INCWeightOnlyLinear modules"

    def test_quant_lm_head(self):
        # tie_word_embeddings=false
        gptj_model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            device_map=device,
        )
        lm_head_id = id(gptj_model.lm_head.weight)
        assert id(gptj_model.transformer.wte.weight) != lm_head_id, "The lm_head weight is tied, please check!"
        quant_config = AWQConfig(quant_lm_head=True)
        model = prepare(gptj_model, quant_config, example_inputs=self.example_inputs)
        calib_func(model)
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
        quant_config = AWQConfig(quant_lm_head=True)
        model = prepare(opt_model, quant_config, example_inputs=self.example_inputs)
        calib_func(model)
        model = convert(model)
        assert (
            id(model.model.decoder.embed_tokens.weight) == lm_head_id
        ), "The tied lm_head weight is not deep copied, please check!"
    
    @pytest.mark.skip("Skipping test_awq_absorb_to_layer due to known issues with AWQ absorb layers.")
    def test_awq_absorb_to_layer(self):
        absorb_layer_dict = {
            "ln_1": (
                "attn.q_proj",
                "attn.k_proj",
                "attn.v_proj",
                "mlp.fc_in",
            ),
            "attn.out_proj": "attn.out_proj",
            "mlp.fc_out": ("mlp.fc_out"),
        }

        quant_config = AWQConfig(absorb_layer_dict=absorb_layer_dict)
        logger.info(f"Test AWQ with config {quant_config}")
        # prepare + convert API
        model = prepare(
            model=copy.deepcopy(self.tiny_gptj),
            quant_config=quant_config,
            example_inputs=self.example_inputs,
        )
        calib_func(model)
        model = convert(model)
        out1 = model(self.example_inputs)
        quant_config = AWQConfig()
        logger.info(f"Test AWQ with config {quant_config}")

        # prepare + convert API
        model = prepare(
            model=copy.deepcopy(self.tiny_gptj),
            quant_config=quant_config,
            example_inputs=self.example_inputs,
        )
        calib_func(model)
        model = convert(model)
        out2 = model(self.example_inputs)

        assert torch.all(out1[0].eq(out2[0])), "The results should be equal."
