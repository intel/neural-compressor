import copy
import unittest

import torch
import transformers

from neural_compressor.common import Logger

logger = Logger().get_logger()
from neural_compressor.torch.quantization import AWQConfig, get_default_awq_config, quantize


def get_gpt_j():
    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        torchscript=True,
    )
    return tiny_gptj


class TestAWQ(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.example_inputs = torch.randn([1, 32])
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

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
        )

        class LLMCalibDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(2):
                    yield example_inputs

        out1 = model(example_inputs)
        quant_config = AWQConfig(bits=8, group_size=-1)
        logger.info(f"Test AWQ with config {quant_config}")
        qdq_model = quantize(model=model, quant_config=quant_config, run_args=LLMCalibDataloader())
        out2 = qdq_model(example_inputs)
        # output data is up to 4, so use big atol=0.5
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=0.5))

        def calib_func(model):
            for i in range(2):
                model(self.lm_input)

        out1 = self.gptj(example_inputs)
        logger.info(f"Test AWQ with config {quant_config}")
        qdq_model = quantize(
            model=self.gptj, quant_config=quant_config, example_inputs=self.lm_input, run_fn=calib_func
        )
        out2 = qdq_model(example_inputs)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-2))

        # default awq_quantize is 4 bits, 32 group size, use big atol=1e-1
        quant_config = AWQConfig()
        logger.info(f"Test AWQ with config {quant_config}")
        qdq_model = quantize(
            model=self.gptj, quant_config=quant_config, example_inputs=self.lm_input, run_fn=calib_func
        )
        out2 = qdq_model(example_inputs)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-1))


if __name__ == "__main__":
    unittest.main()
