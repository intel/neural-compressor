import copy
import unittest

import torch
import transformers

from neural_compressor.common import Logger

logger = Logger().get_logger()
from neural_compressor.torch.quantization import AWQConfig, convert, get_default_awq_config, prepare, quantize


def get_gpt_j():
    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        torchscript=True,
    )
    return tiny_gptj


class TestAWQ(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.lm_input = torch.ones([1, 10], dtype=torch.long)
        self.gptj = get_gpt_j()

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestAWQ test: {self.id()}")

    def test_awq(self):
        example_inputs = torch.ones([1, 10], dtype=torch.long)

        @torch.no_grad()
        def calib_func(model):
            for i in range(2):
                model(self.lm_input)

        out1 = self.gptj(example_inputs)
        quant_config = AWQConfig(bits=8, group_size=-1)
        logger.info(f"Test AWQ with config {quant_config}")
        model = prepare(
            model=self.gptj,
            quant_config=quant_config,
            example_inputs=self.lm_input,
        )
        calib_func(model)
        qdq_model = convert(model)
        out2 = qdq_model(example_inputs)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-2))

        # default awq_quantize is 4 bits, 32 group size, use big atol=1e-1
        quant_config = AWQConfig()
        logger.info(f"Test AWQ with config {quant_config}")
        model = prepare(
            model=self.gptj,
            quant_config=quant_config,
            example_inputs=self.lm_input,
        )
        calib_func(model)
        qdq_model = convert(model)
        out2 = qdq_model(example_inputs)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-1))

    def test_awq_with_quantize_API(self):
        example_inputs = torch.ones([1, 10], dtype=torch.long)

        @torch.no_grad()
        def calib_func(model):
            for i in range(2):
                model(self.lm_input)

        quant_config = get_default_awq_config()
        logger.info(f"Test AWQ with config {quant_config}")

        # prepare + convert API
        model = prepare(
            model=copy.deepcopy(self.gptj),
            quant_config=quant_config,
            example_inputs=self.lm_input,
        )
        calib_func(model)
        qdq_model = convert(model)
        out1 = qdq_model(example_inputs)

        # quantize API
        qdq_model = quantize(
            model=copy.deepcopy(self.gptj), quant_config=quant_config, example_inputs=self.lm_input, run_fn=calib_func
        )
        out2 = qdq_model(example_inputs)

        # compare the results of calling `convert` + `prepare` and calling `quantize`
        assert torch.all(
            out1[0].eq(out2[0])
        ), "The results of calling `convert` + `prepare` and calling `quantize` should be equal."


if __name__ == "__main__":
    unittest.main()
