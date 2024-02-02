import copy
import unittest

import torch

from neural_compressor.common import Logger

logger = Logger().get_logger()
from neural_compressor.torch.quantization import AWQConfig, get_default_awq_config, quantize


def get_gpt_j():
    import transformers

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
        example_inputs = self.example_inputs

        class LLMCalibDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(2):
                    yield example_inputs

        example_inputs = torch.ones([1, 10], dtype=torch.long)

        quant_config = AWQConfig(bits=8, group_size=-1)
        # quant_config.set_local("lm_head", AWQConfig(dtype="fp32"))
        logger.info(f"Test AWQ with config {quant_config}")

        def calib_func(model):
            for i in range(2):
                model(self.lm_input)

        out1 = self.gptj(example_inputs)
        qdq_model = quantize(
            model=self.gptj, quant_config=quant_config, example_inputs=self.lm_input, run_fn=calib_func
        )

        out2 = qdq_model(example_inputs)
        print(out1)
        print(out2)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-2))
        ###


if __name__ == "__main__":
    unittest.main()
