import copy

import pytest
import torch
import transformers

from neural_compressor.torch.quantization import GPTQConfig, get_default_gptq_config, get_default_rtn_config, quantize
from neural_compressor.torch.quantization.modules import WeightOnlyLinear


@pytest.mark.xfail(raises=ValueError)  # GPTQ uses ValueError to collect input data of the first block
class TestGPTQQuant:
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

    def test_accuracy_improvement(self):
        model = copy.deepcopy(self.tiny_gptj)

        def run_fn(model):
            model(self.example_inputs)

        gptq_config = get_default_gptq_config()
        model = quantize(model, gptq_config, run_fn=run_fn)
        out = model(self.example_inputs)[0]
        print((out - self.label).amax())
        print((self.q_label - self.label).amax())

    # @pytest.mark.parametrize(
    #     "bits, use_sym, group_size, group_dim",
    #     [
    #         (8, True, 128, 1),
    #         (4, True, 128, 1),
    #         (4, False, 32, 1),
    #         (4, True, 32, 0),
    #         (4, False, -1, 1),
    #         (2, True, 8, 1),
    #     ],
    # )
    # def test_int_params(self, bits, use_sym, group_size, group_dim):
    #     model = copy.deepcopy(self.tiny_gptj)
    #     quant_config = GPTQConfig(
    #         bits=bits,
    #         use_sym=use_sym,
    #         group_size=group_size,
    #         group_dim=group_dim,
    #         act_order=False,
    #         block_size=1024,
    #     )
    #     model = quantize(model, quant_config)
    #     out = model(self.example_inputs)[0]
    #     assert (out != self.label).all(), "WOQ output should be different with raw output"
    #     if (bits, use_sym, group_size, group_dim) == (8, True, 128, 1):
    #         assert torch.allclose(out, self.label, atol=0.01), "Accuracy gap atol > 0.01 is unexpected."
    #     if (bits, use_sym, group_size, group_dim) == [(4, True, 128, 0), (4, True, 32, 1)]:
    #         assert torch.allclose(out, self.label, atol=0.1), "Accuracy gap atol > 0.1 is unexpected."
    #     if (bits, use_sym, group_size, group_dim) == [(4, False, 32, 0), (4, False, -1, 1), (2, True, 16, 1)]:
    #         assert torch.allclose(out, self.label, atol=0.5), "Accuracy gap atol > 0.5 is unexpected."
