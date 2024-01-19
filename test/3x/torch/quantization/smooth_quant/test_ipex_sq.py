import unittest

import torch

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()

try:
    import intel_extension_for_pytorch as ipex

    TEST_IPEX = True
except:
    TEST_IPEX = False

assert TEST_IPEX, "Please install intel extension for pytorch"


class TestSQ(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        logger.info(f"Running TestSQ test: {self.id()}")

    def test_sq_config(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, False)

            def forward(self, x):
                x = self.linear(x)
                x = x + x
                return x

        example_input = torch.tensor([[torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max]])
        model = M()
        model.linear.weight = torch.nn.Parameter(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

        def calib_func(model):
            model(example_input)

        from neural_compressor.torch import SmoothQuantConfig, quantize

        quant_config = SmoothQuantConfig(act_algo="minmax")
        q_model = quantize(
            model=model,
            quant_config=quant_config,
            # example_input = example_input,
            run_fn=calib_func,
            inplace=True,
        )


if __name__ == "__main__":
    unittest.main()
