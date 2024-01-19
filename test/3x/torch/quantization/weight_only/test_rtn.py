import unittest

import torch

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()


def get_gpt_j():
    import transformers

    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        torchscript=True,
    )
    return tiny_gptj


def build_simple_torch_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = torch.nn.Linear(8, 30)
            self.fc2 = torch.nn.Linear(30, 60)
            self.fc3 = torch.nn.Linear(60, 30)
            self.fc4 = torch.nn.Linear(30, 50)

        def forward(self, x):
            out = self.fc1(x)
            out = self.fc2(out)
            out = self.fc3(out)
            out = self.fc4(out)
            return out

    model = Model()
    return model


class TestRTNQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestRTNQuant test: {self.id()}")

    def _apply_rtn(self, quant_config):
        logger.info(f"Test RTN with config {quant_config}")
        from neural_compressor.torch import quantize

        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)
        return qmodel

    def test_rtn(self):
        from neural_compressor.torch import RTNConfig

        # some tests were skipped to accelerate the CI
        rnt_options = {
            "weight_dtype": ["int", "int8", "nf4", "fp4_e2m1_bnb"],
            "weight_bits": [4, 1, 8],
            "weight_group_size": [32, -1, 1, 512],
            "weight_sym": [True, False],
            "act_dtype": ["fp32"],
            "enable_full_range": [False, True],
            "enable_mse_search": [False],
            "group_dim": [1, 0],
            "return_int": [False, True],
        }
        from itertools import product

        keys = RTNConfig.params_list
        for value in product(*rnt_options.values()):
            d = dict(zip(keys, value))
            if (d["weight_dtype"] == "int" and d["weight_bits"] != 8) or (
                d["enable_full_range"]
                and d["enable_mse_search"]
                or (d["return_int"] and (d["group_dim"] != 1 or d["weight_bits"] != 8))
            ):
                continue
            quant_config = RTNConfig(**d)
            self._apply_rtn(quant_config)

    def test_rtn_return_type(self):
        from neural_compressor.torch import RTNConfig

        for return_int in [True, False]:
            quant_config = RTNConfig(return_int=return_int)
            qmodel = self._apply_rtn(quant_config)

    def test_rtn_mse_search(self):
        from neural_compressor.torch import RTNConfig

        quant_config = RTNConfig(enable_mse_search=True)
        qmodel = self._apply_rtn(quant_config)

    def test_rtn_recover(self):
        from neural_compressor.torch import RTNConfig

        quant_config = RTNConfig(return_int=True)
        qmodel = self._apply_rtn(quant_config)
        input = torch.randn(4, 8)
        # test forward
        out = qmodel(input)
        recovered_fc1 = qmodel.fc1.recover()
        self.assertIsNotNone(recovered_fc1)

    def test_weight_only_linear(self):
        from neural_compressor.torch.algorithms.weight_only.rtn import rtn_quantize

        model = build_simple_torch_model()
        options = {
            "compression_dtype": [torch.int8, torch.int16, torch.int32, torch.int64],
            "compression_dim": [0, 1],
            "module": [model.fc1, model.fc2, model.fc3, model.fc4],
        }
        from itertools import product

        for compression_dtype, compression_dim, module in product(*options.values()):
            q_model = rtn_quantize(
                model=module,
                return_int=True,
                compression_dtype=compression_dtype,
                compression_dim=compression_dim,
            )


if __name__ == "__main__":
    unittest.main()
