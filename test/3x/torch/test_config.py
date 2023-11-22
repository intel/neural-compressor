import unittest

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()
import torch


def build_simple_torch_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = torch.nn.Linear(30, 50)
            self.fc2 = torch.nn.Linear(50, 30)
            self.fc3 = torch.nn.Linear(30, 5)

        def forward(self, x):
            out = self.fc1(x)
            out = self.fc2(out)
            out = self.fc3(out)
            return out

    model = Model()
    return model


class TestQuantizationConfig(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fp32_model = build_simple_torch_model()

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestQuantizationConfig test: {self.id()}")

    def test_quantize_rtn_from_dict_default(self):
        logger.info("test_quantize_rtn_from_dict_default")
        from neural_compressor.torch import get_default_rtn_config, quantize

        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config=get_default_rtn_config())
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_dict_beginner(self):
        from neural_compressor.torch import quantize

        quant_config = {
            "rtn_weight_only_quant": {
                "weight_dtype": "nf4",
                "weight_bits": 4,
                "weight_group_size": 32,
            },
        }
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_class_beginner(self):
        from neural_compressor.torch import RTNWeightQuantConfig, quantize

        quant_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4", weight_group_size=32)
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_dict_advance(self):
        from neural_compressor.torch import quantize

        fp32_model = build_simple_torch_model()
        quant_config = {
            "rtn_weight_only_quant": {
                "global": {
                    "weight_dtype": "nf4",
                    "weight_bits": 4,
                    "weight_group_size": 32,
                },
                "local": {
                    "fc1": {
                        "weight_dtype": "int8",
                        "weight_bits": 4,
                    }
                },
            }
        }
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_class_advance(self):
        from neural_compressor.torch import RTNWeightQuantConfig, quantize

        quant_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4")
        # set operator instance
        fc1_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="int8")
        quant_config.set_local("model.fc1", fc1_config)
        # get model and quantize
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_config_from_dict(self):
        from neural_compressor.torch import RTNWeightQuantConfig

        quant_config = {
            "rtn_weight_only_quant": {
                "global": {
                    "weight_dtype": "nf4",
                    "weight_bits": 4,
                    "weight_group_size": 32,
                },
                "local": {
                    "fc1": {
                        "weight_dtype": "int8",
                        "weight_bits": 4,
                    }
                },
            }
        }
        config = RTNWeightQuantConfig.from_dict(quant_config)
        self.assertIsNotNone(config.local_config)

    def test_config_to_dict(self):
        from neural_compressor.torch import RTNWeightQuantConfig

        quant_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4")
        fc1_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="int8")
        quant_config.set_local("model.fc1", fc1_config)
        config_dict = quant_config.to_dict()
        self.assertIn("global", config_dict)
        self.assertIn("local", config_dict)

    def test_same_type_configs_addition(self):
        from neural_compressor.torch import RTNWeightQuantConfig

        quant_config1 = {
            "rtn_weight_only_quant": {
                "weight_dtype": "nf4",
                "weight_bits": 4,
                "weight_group_size": 32,
            },
        }
        q_config = RTNWeightQuantConfig.from_dict(quant_config1["rtn_weight_only_quant"])
        quant_config2 = {
            "rtn_weight_only_quant": {
                "global": {
                    "weight_bits": 8,
                    "weight_group_size": 32,
                },
                "local": {
                    "fc1": {
                        "weight_dtype": "int8",
                        "weight_bits": 4,
                    }
                },
            }
        }
        q_config2 = RTNWeightQuantConfig.from_dict(quant_config2["rtn_weight_only_quant"])
        q_config3 = q_config + q_config2
        q3_dict = q_config3.to_dict()
        for op_name, op_config in quant_config2["rtn_weight_only_quant"]["local"].items():
            for attr, val in op_config.items():
                self.assertEqual(q3_dict["local"][op_name][attr], val)
        self.assertNotEqual(
            q3_dict["global"]["weight_bits"], quant_config2["rtn_weight_only_quant"]["global"]["weight_bits"]
        )

    def test_diff_types_configs_addition(self):
        from neural_compressor.torch import DummyConfig, RTNWeightQuantConfig

        quant_config1 = {
            "rtn_weight_only_quant": {
                "weight_dtype": "nf4",
                "weight_bits": 4,
                "weight_group_size": 32,
            },
        }
        q_config = RTNWeightQuantConfig.from_dict(quant_config1["rtn_weight_only_quant"])
        d_config = DummyConfig(act_dtype="fp32", dummy_attr=3)
        combined_config = q_config + d_config
        combined_config_d = combined_config.to_dict()
        logger.info(combined_config)
        self.assertTrue("rtn_weight_only_quant" in combined_config_d)
        self.assertIn("dummy_config", combined_config_d)

    def test_composable_config_addition(self):
        from neural_compressor.torch import DummyConfig, RTNWeightQuantConfig

        quant_config1 = {
            "rtn_weight_only_quant": {
                "weight_dtype": "nf4",
                "weight_bits": 4,
                "weight_group_size": 32,
            },
        }
        q_config = RTNWeightQuantConfig.from_dict(quant_config1["rtn_weight_only_quant"])
        d_config = DummyConfig(act_dtype="fp32", dummy_attr=3)
        combined_config = q_config + d_config
        combined_config_d = combined_config.to_dict()
        logger.info(combined_config)
        self.assertTrue("rtn_weight_only_quant" in combined_config_d)
        self.assertIn("dummy_config", combined_config_d)
        combined_config2 = combined_config + d_config
        combined_config3 = combined_config + combined_config2

    def test_config_mapping(self):
        from neural_compressor.torch import RTNWeightQuantConfig
        from neural_compressor.torch.utils import get_model_info

        quant_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4")
        # set operator instance
        fc1_config = RTNWeightQuantConfig(weight_bits=6, weight_dtype="int8")
        quant_config.set_local("fc1", fc1_config)
        # get model and quantize
        fp32_model = build_simple_torch_model()
        model_info = get_model_info(fp32_model, white_module_list=[torch.nn.Linear])
        logger.info(quant_config)
        configs_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.info(configs_mapping)
        self.assertTrue(configs_mapping[(torch.nn.Linear, "fc1")].weight_bits == 6)
        self.assertTrue(configs_mapping[(torch.nn.Linear, "fc2")].weight_bits == 4)


if __name__ == "__main__":
    unittest.main()
