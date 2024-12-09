import copy
import unittest

import pytest
import torch
import transformers

import neural_compressor.torch.utils as torch_utils
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    AWQConfig,
    FP8Config,
    GPTQConfig,
    HQQConfig,
    INT8StaticQuantConfig,
    RTNConfig,
    SmoothQuantConfig,
    StaticQuantConfig,
    TEQConfig,
    get_default_AutoRound_config,
    get_default_gptq_config,
    get_default_hqq_config,
    get_default_rtn_config,
    quantize,
)
from neural_compressor.torch.utils import get_model_info, logger


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
        self.input = torch.randn(1, 30)
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )
        self.lm_input = torch.ones([1, 10], dtype=torch.long)

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestQuantizationConfig test: {self.id()}")

    def test_quantize_rtn_from_dict_default(self):
        logger.info("test_quantize_rtn_from_dict_default")

        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config=get_default_rtn_config())
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_dict_beginner(self):
        quant_config = {
            "rtn": {
                "dtype": "nf4",
                "bits": 4,
                "group_size": 32,
            },
        }
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_class_beginner(self):
        quant_config = RTNConfig(bits=4, dtype="nf4", group_size=32)
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtndq_from_class_beginner(self):
        fp32_config = RTNConfig(dtype="fp32")

        fp32_model = copy.deepcopy(self.gptj)
        quant_config = RTNConfig(
            bits=4,
            dtype="int",
            use_sym=False,
            group_size=32,
        )
        quant_config.set_local("lm_head", fp32_config)
        qmodel = quantize(fp32_model, quant_config)
        out2 = qmodel(self.lm_input)

        fp32_model = copy.deepcopy(self.gptj)

    def test_quantize_rtn_from_dict_advance(self):
        fp32_model = build_simple_torch_model()
        quant_config = {
            "rtn": {
                "global": {
                    "dtype": "nf4",
                    "bits": 4,
                    "group_size": 32,
                },
                "local": {
                    "fc1": {
                        "dtype": "int8",
                        "bits": 4,
                    }
                },
            }
        }
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_class_advance(self):
        quant_config = RTNConfig(bits=4, dtype="nf4")
        # set operator instance
        fc1_config = RTNConfig(bits=4, dtype="int8")
        quant_config.set_local("model.fc1", fc1_config)
        # get model and quantize
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_config_white_lst(self):
        global_config = RTNConfig(bits=4, dtype="nf4")
        # set operator instance
        fc1_config = RTNConfig(bits=4, dtype="int8", white_list=["model.fc1"])
        # get model and quantize
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config=global_config + fc1_config)
        self.assertIsNotNone(qmodel)

    def test_config_white_lst2(self):
        global_config = RTNConfig(bits=4, dtype="nf4")
        # set operator instance
        fc1_config = RTNConfig(bits=6, dtype="int8", white_list=["fc1"])
        quant_config = global_config + fc1_config
        # get model and quantize
        fp32_model = build_simple_torch_model()
        model_info = get_model_info(fp32_model, white_module_list=[torch.nn.Linear])
        logger.info(quant_config)
        configs_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.info(configs_mapping)
        self.assertTrue(configs_mapping[("fc1", "Linear")].bits == 6)
        self.assertTrue(configs_mapping[("fc2", "Linear")].bits == 4)

    def test_config_from_dict(self):
        quant_config = {
            "rtn": {
                "global": {
                    "dtype": "nf4",
                    "bits": 4,
                    "group_size": 32,
                },
                "local": {
                    "fc1": {
                        "dtype": "int8",
                        "bits": 4,
                    }
                },
            }
        }
        config = RTNConfig.from_dict(quant_config["rtn"])
        self.assertIsNotNone(config.local_config)

    def test_config_to_dict(self):
        quant_config = RTNConfig(bits=4, dtype="nf4")
        fc1_config = RTNConfig(bits=4, dtype="int8")
        quant_config.set_local("model.fc1", fc1_config)
        config_dict = quant_config.to_dict()
        self.assertIn("global", config_dict)
        self.assertIn("local", config_dict)

    def test_same_type_configs_addition(self):
        quant_config1 = {
            "rtn": {
                "dtype": "nf4",
                "bits": 4,
                "group_size": 32,
            },
        }
        q_config = RTNConfig.from_dict(quant_config1["rtn"])
        quant_config2 = {
            "rtn": {
                "global": {
                    "bits": 8,
                    "group_size": 32,
                },
                "local": {
                    "fc1": {
                        "dtype": "int8",
                        "bits": 4,
                    }
                },
            }
        }
        q_config2 = RTNConfig.from_dict(quant_config2["rtn"])
        q_config3 = q_config + q_config2
        q3_dict = q_config3.to_dict()
        for op_name, op_config in quant_config2["rtn"]["local"].items():
            for attr, val in op_config.items():
                self.assertEqual(q3_dict["local"][op_name][attr], val)
        self.assertNotEqual(q3_dict["global"]["bits"], quant_config2["rtn"]["global"]["bits"])

    def test_diff_types_configs_addition(self):
        quant_config1 = {
            "rtn": {
                "dtype": "nf4",
                "bits": 4,
                "group_size": 32,
            },
        }
        q_config = RTNConfig.from_dict(quant_config1["rtn"])
        d_config = GPTQConfig(double_quant_bits=4)
        combined_config = q_config + d_config
        combined_config_d = combined_config.to_dict()
        logger.info(combined_config)
        self.assertTrue("rtn" in combined_config_d)
        self.assertIn("gptq", combined_config_d)

    def test_composable_config_addition(self):
        quant_config1 = {
            "rtn": {
                "dtype": "nf4",
                "bits": 4,
                "group_size": 32,
            },
        }
        q_config = RTNConfig.from_dict(quant_config1["rtn"])
        d_config = GPTQConfig(double_quant_bits=4)
        combined_config = q_config + d_config
        combined_config_d = combined_config.to_dict()
        logger.info(combined_config)
        self.assertTrue("rtn" in combined_config_d)
        self.assertIn("gptq", combined_config_d)
        combined_config2 = combined_config + d_config
        combined_config3 = combined_config + combined_config2

    def test_config_mapping(self):
        quant_config = RTNConfig(bits=4, dtype="nf4")
        # set operator instance
        fc1_config = RTNConfig(bits=6, dtype="int8")
        quant_config.set_local("fc1", fc1_config)
        # get model and quantize
        fp32_model = build_simple_torch_model()
        model_info = get_model_info(fp32_model, white_module_list=[torch.nn.Linear])
        logger.info(quant_config)
        configs_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.info(configs_mapping)
        self.assertTrue(configs_mapping[("fc1", "Linear")].bits == 6)
        self.assertTrue(configs_mapping[("fc2", "Linear")].bits == 4)
        # test regular matching
        fc_config = RTNConfig(bits=5, dtype="int8")
        quant_config.set_local("fc", fc_config)
        configs_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.info(configs_mapping)
        self.assertTrue(configs_mapping[("fc1", "Linear")].bits == 5)
        self.assertTrue(configs_mapping[("fc2", "Linear")].bits == 5)
        self.assertTrue(configs_mapping[("fc3", "Linear")].bits == 5)

    def test_set_local_op_type(self):
        quant_config = RTNConfig(bits=4, dtype="nf4")
        # set all `Linear`
        fc1_config = RTNConfig(bits=6, dtype="int8")
        quant_config.set_local(torch.nn.Linear, fc1_config)
        # get model and quantize
        fp32_model = build_simple_torch_model()
        model_info = get_model_info(fp32_model, white_module_list=[torch.nn.Linear])
        logger.info(quant_config)
        configs_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.info(configs_mapping)
        self.assertTrue(configs_mapping[("fc1", "Linear")].bits == 6)
        self.assertTrue(configs_mapping[("fc2", "Linear")].bits == 6)
        self.assertTrue(configs_mapping[("fc3", "Linear")].bits == 6)

    def test_gptq_config(self):
        gptq_config1 = GPTQConfig(bits=8, act_order=True)
        quant_config_dict = {
            "gptq": {"bits": 8, "act_order": True},
        }
        gptq_config2 = GPTQConfig.from_dict(quant_config_dict["gptq"])
        self.assertEqual(gptq_config1.to_dict(), gptq_config2.to_dict())

    def test_awq_config(self):
        awq_config1 = AWQConfig(bits=8, use_auto_scale=True, folding=False)
        quant_config_dict = {
            "awq": {"bits": 8, "use_auto_scale": True, "folding": False},
        }
        awq_config2 = AWQConfig.from_dict(quant_config_dict["awq"])
        self.assertEqual(awq_config1.to_dict(), awq_config2.to_dict())

    def test_teq_config(self):
        absorb_dict = {"transformer.h.0.mlp.fc_in": ["transformer.h.0.mlp.fc_out"]}
        teq_config1 = TEQConfig(bits=8, absorb_to_layer=absorb_dict, folding=False)
        quant_config_dict = {
            "teq": {"bits": 8, "absorb_to_layer": absorb_dict, "folding": False},
        }
        teq_config2 = TEQConfig.from_dict(quant_config_dict["teq"])
        self.assertEqual(teq_config1.to_dict(), teq_config2.to_dict())

    def test_autoround_config(self):
        autoround_config1 = AutoRoundConfig(batch_size=16, scale_dtype="fp32")
        quant_config_dict = {
            "autoround": {"batch_size": 16, "scale_dtype": "fp32"},
        }
        autoround_config2 = AutoRoundConfig.from_dict(quant_config_dict["autoround"])
        self.assertEqual(autoround_config1.to_dict(), autoround_config2.to_dict())

    def test_static_quant_config(self):
        static_config1 = StaticQuantConfig(w_dtype="int8", act_sym=True, act_algo="minmax")
        quant_config_dict = {"static": {"w_dtype": "int8", "act_sym": True, "act_algo": "minmax"}}
        static_config2 = StaticQuantConfig.from_dict(quant_config_dict["static"])
        self.assertEqual(static_config1.to_dict(), static_config2.to_dict())

    def test_smooth_quant_config(self):
        sq_config1 = SmoothQuantConfig(alpha=0.8, folding=True)
        quant_config_dict = {"sq": {"alpha": 0.8, "folding": True}}
        sq_config2 = SmoothQuantConfig.from_dict(quant_config_dict["sq"])
        self.assertEqual(sq_config1.to_dict(), sq_config2.to_dict())

    def test_hqq_config(self):
        hqq_config = HQQConfig(bits=4, group_size=64, quant_zero=True)
        quant_config_dict = {"hqq": {"bits": 4, "group_size": 64, "quant_zero": True}}
        hqq_config2 = HQQConfig.from_dict(quant_config_dict["hqq"])
        self.assertEqual(hqq_config.to_dict(), hqq_config2.to_dict())


class TestQuantConfigBasedonProcessorType:

    @pytest.mark.parametrize("config_cls", [RTNConfig, GPTQConfig, AutoRoundConfig])
    def test_get_config_based_on_processor_type(self, config_cls):
        config_for_client = config_cls.get_predefined_configs()[torch_utils.ProcessorType.Client]
        assert (
            config_for_client.use_layer_wise
        ), f"Expect use_layer_wise to be True, got {config_for_client.use_layer_wise}"

        config_for_server = config_cls.get_predefined_configs()[torch_utils.ProcessorType.Server]
        assert (
            config_for_server.use_layer_wise is False
        ), f"Expect use_layer_wise to be False, got {config_for_server.use_layer_wise}"

    @pytest.fixture
    def force_server(self, monkeypatch):
        monkeypatch.setattr(torch_utils.utility.cpu_info, "sockets", 2)

    def test_get_default_config_force_server(self, force_server):
        rtn_config = get_default_rtn_config()
        assert not rtn_config.use_layer_wise, f"Expect use_layer_wise to be `False`, got {rtn_config.use_layer_wise}"
        gptq_config = get_default_gptq_config()
        assert not gptq_config.use_layer_wise, f"Expect use_layer_wise to be `False`, got {gptq_config.use_layer_wise}"

    @pytest.mark.parametrize("p_type", [None, torch_utils.ProcessorType.Client, torch_utils.ProcessorType.Server])
    def test_get_default_config(self, p_type):
        rtn_config = get_default_rtn_config(processor_type=p_type)
        assert rtn_config.use_layer_wise == (
            p_type == torch_utils.ProcessorType.Client
        ), f"Expect use_layer_wise to be {p_type == torch_utils.ProcessorType.Client}, got {rtn_config.use_layer_wise}"
        gptq_config = get_default_gptq_config(processor_type=p_type)
        assert gptq_config.use_layer_wise == (
            p_type == torch_utils.ProcessorType.Client
        ), f"Expect use_layer_wise to be {p_type == torch_utils.ProcessorType.Client}, got {gptq_config.use_layer_wise}"
        autoround_config = get_default_AutoRound_config(processor_type=p_type)
        assert autoround_config.use_layer_wise == (
            p_type == torch_utils.ProcessorType.Client
        ), f"Expect use_layer_wise to be {p_type == torch_utils.ProcessorType.Client}, got {autoround_config.use_layer_wise}"


def test_auto_config_mapping():
    # case 1
    class_obj = StaticQuantConfig()
    assert isinstance(
        class_obj, INT8StaticQuantConfig
    ), "StaticQuantConfig should be mapped to INT8StaticQuantConfig by default."
    # case 2
    class_obj = StaticQuantConfig(fp8_config="E4M3")
    assert isinstance(class_obj, FP8Config), "StaticQuantConfig should be mapped to FP8Config with fp8_config argument."
    # case 3
    class_obj = StaticQuantConfig(fp8_config="E4M3", observer="maxabs")
    assert isinstance(class_obj, FP8Config), "StaticQuantConfig should be mapped to FP8Config with fp8_config argument."
    # case 4
    class_obj = StaticQuantConfig(act_sym=True, act_algo="kl")
    assert isinstance(class_obj, INT8StaticQuantConfig), "StaticQuantConfig should be mapped to INT8StaticQuantConfig."
