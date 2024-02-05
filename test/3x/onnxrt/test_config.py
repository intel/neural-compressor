import copy
import os
import shutil
import unittest

import numpy as np
import onnx
from optimum.exporters.onnx import main_export

from neural_compressor.common import Logger

logger = Logger().get_logger()


def find_onnx_file(folder_path):
    # return first .onnx file path in folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".onnx"):
                return os.path.join(root, file)
    return None


def build_simple_onnx_model():
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 5, 5])
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [1, 5, 2])
    D = onnx.helper.make_tensor_value_info("D", onnx.TensorProto.FLOAT, [1, 5, 2])
    H = onnx.helper.make_tensor_value_info("H", onnx.TensorProto.FLOAT, [1, 5, 2])

    e_value = np.random.randint(2, size=(10)).astype(np.float32)
    B_init = onnx.helper.make_tensor("B", onnx.TensorProto.FLOAT, [5, 2], e_value.reshape(10).tolist())
    E_init = onnx.helper.make_tensor("E", onnx.TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())

    matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"], name="Matmul")
    add = onnx.helper.make_node("Add", ["C", "E"], ["D"], name="add")

    f_value = np.random.randint(2, size=(10)).astype(np.float32)
    F_init = onnx.helper.make_tensor("F", onnx.TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())
    add2 = onnx.helper.make_node("Add", ["D", "F"], ["H"], name="add2")

    graph = onnx.helper.make_graph([matmul_node, add, add2], "test_graph_1", [A], [H], [B_init, E_init, F_init])
    model = onnx.helper.make_model(graph)
    model = onnx.helper.make_model(graph, **{"opset_imports": [onnx.helper.make_opsetid("", 13)]})
    return model


class TestQuantizationConfig(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        main_export(
            "hf-internal-testing/tiny-random-gptj",
            output="gptj",
        )
        self.gptj = find_onnx_file("./gptj")

        simple_onnx_model = build_simple_onnx_model()
        onnx.save(simple_onnx_model, "simple_onnx_model.onnx")
        self.simple_onnx_model = "simple_onnx_model.onnx"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("gptj", ignore_errors=True)
        os.remove("simple_onnx_model.onnx")

    def setUp(self):
        # print the test name
        logger.info(f"Running TestQuantizationConfig test: {self.id()}")

    def _check_node_is_quantized(self, model, node_name):
        for node in model.graph.node:
            if (node.name == node_name or node.name == node_name + "_Q4") and node.op_type in [
                "MatMulNBits",
                "MatMulFpQ4",
            ]:
                return True
        return False

    def _count_woq_matmul(self, q_model, bits=4, group_size=32):
        op_names = [
            i.name
            for i in q_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(bits, group_size))
        ]
        return len(op_names)

    def test_config_white_lst(self):
        from neural_compressor.onnxrt import RTNConfig
        from neural_compressor.onnxrt.quantization.quantize import _quantize

        global_config = RTNConfig(weight_bits=4)
        # set operator instance
        fc_out_config = RTNConfig(weight_dtype="fp32", white_list=["/h.4/mlp/fc_out/MatMul"])
        # get model and quantize
        fp32_model = self.gptj
        qmodel = _quantize(fp32_model, quant_config=global_config + fc_out_config)
        self.assertIsNotNone(qmodel)
        self.assertEqual(self._count_woq_matmul(qmodel), 29)
        self.assertFalse(self._check_node_is_quantized(qmodel, "/h.4/mlp/fc_out/MatMul"))

    def test_config_white_lst2(self):
        from neural_compressor.onnxrt import RTNConfig
        from neural_compressor.onnxrt.quantization.quantize import _quantize

        global_config = RTNConfig(weight_dtype="fp32")
        # set operator instance
        fc_out_config = RTNConfig(weight_bits=4, white_list=["/h.4/mlp/fc_out/MatMul"])
        # get model and quantize
        fp32_model = self.gptj
        qmodel = _quantize(fp32_model, quant_config=global_config + fc_out_config)
        self.assertIsNotNone(qmodel)
        self.assertEqual(self._count_woq_matmul(qmodel), 1)
        self.assertTrue(self._check_node_is_quantized(qmodel, "/h.4/mlp/fc_out/MatMul"))

    def test_config_white_lst3(self):
        from neural_compressor.onnxrt import RTNConfig
        from neural_compressor.onnxrt.utils.utility import get_model_info

        global_config = RTNConfig(weight_bits=4)
        # set operator instance
        fc_out_config = RTNConfig(weight_bits=8, white_list=["/h.4/mlp/fc_out/MatMul"])
        quant_config = global_config + fc_out_config
        # get model and quantize
        fp32_model = self.gptj
        model_info = get_model_info(fp32_model, white_op_type_list=["MatMul"])
        logger.info(quant_config)
        configs_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.info(configs_mapping)
        self.assertTrue(configs_mapping[("/h.4/mlp/fc_out/MatMul", "MatMul")].weight_bits == 8)
        self.assertTrue(configs_mapping[("/h.4/mlp/fc_in/MatMul", "MatMul")].weight_bits == 4)

    def test_config_from_dict(self):
        from neural_compressor.onnxrt import RTNConfig

        quant_config = {
            "rtn": {
                "global": {
                    "weight_dtype": "int",
                    "weight_bits": 4,
                    "weight_group_size": 32,
                },
                "local": {
                    "fc1": {
                        "weight_dtype": "int",
                        "weight_bits": 8,
                    }
                },
            }
        }
        config = RTNConfig.from_dict(quant_config["rtn"])
        self.assertIsNotNone(config.local_config)

    def test_config_to_dict(self):
        from neural_compressor.onnxrt import RTNConfig

        quant_config = RTNConfig(weight_bits=4)
        fc_out_config = RTNConfig(weight_bits=8)
        quant_config.set_local("/h.4/mlp/fc_out/MatMul", fc_out_config)
        config_dict = quant_config.to_dict()
        self.assertIn("global", config_dict)
        self.assertIn("local", config_dict)

    def test_same_type_configs_addition(self):
        from neural_compressor.onnxrt import RTNConfig

        quant_config1 = {
            "rtn": {
                "weight_dtype": "int",
                "weight_bits": 4,
                "weight_group_size": 32,
            },
        }
        q_config = RTNConfig.from_dict(quant_config1["rtn"])
        quant_config2 = {
            "rtn": {
                "global": {
                    "weight_bits": 8,
                    "weight_group_size": 32,
                },
                "local": {
                    "/h.4/mlp/fc_out/MatMul": {
                        "weight_dtype": "int",
                        "weight_bits": 4,
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
        self.assertNotEqual(q3_dict["global"]["weight_bits"], quant_config2["rtn"]["global"]["weight_bits"])

    def test_config_mapping(self):
        from neural_compressor.onnxrt import RTNConfig
        from neural_compressor.onnxrt.utils.utility import get_model_info

        quant_config = RTNConfig(weight_bits=4)
        # set operator instance
        fc_out_config = RTNConfig(weight_bits=8)
        quant_config.set_local("/h.4/mlp/fc_out/MatMul", fc_out_config)
        # get model and quantize
        fp32_model = self.gptj
        model_info = get_model_info(fp32_model, white_op_type_list=["MatMul"])
        logger.info(quant_config)
        configs_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.info(configs_mapping)
        self.assertTrue(configs_mapping[("/h.4/mlp/fc_out/MatMul", "MatMul")].weight_bits == 8)
        self.assertTrue(configs_mapping[("/h.4/mlp/fc_in/MatMul", "MatMul")].weight_bits == 4)
        # test regular matching
        fc_config = RTNConfig(weight_bits=3)
        quant_config.set_local("/h.[1-4]/mlp/fc_out/MatMul", fc_config)
        configs_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.info(configs_mapping)
        self.assertTrue(configs_mapping[("/h.4/mlp/fc_out/MatMul", "MatMul")].weight_bits == 3)
        self.assertTrue(configs_mapping[("/h.3/mlp/fc_out/MatMul", "MatMul")].weight_bits == 3)
        self.assertTrue(configs_mapping[("/h.2/mlp/fc_out/MatMul", "MatMul")].weight_bits == 3)
        self.assertTrue(configs_mapping[("/h.1/mlp/fc_out/MatMul", "MatMul")].weight_bits == 3)

    def test_diff_types_configs_addition(self):
        from neural_compressor.onnxrt import GPTQConfig, RTNConfig

        quant_config1 = {
            "rtn": {
                "weight_bits": 4,
                "weight_group_size": 32,
            },
        }
        q_config = RTNConfig.from_dict(quant_config1["rtn"])
        d_config = GPTQConfig(weight_group_size=128)
        combined_config = q_config + d_config
        combined_config_d = combined_config.to_dict()
        logger.info(combined_config)
        self.assertIn("rtn", combined_config_d)
        self.assertIn("gptq", combined_config_d)


class TestQuantConfigForAutotune(unittest.TestCase):
    def test_expand_config(self):
        # test the expand functionalities, the user is not aware it
        from neural_compressor.onnxrt import RTNConfig

        tune_config = RTNConfig(weight_bits=[4, 8])
        expand_config_list = RTNConfig.expand(tune_config)
        self.assertEqual(expand_config_list[0].weight_bits, 4)
        self.assertEqual(expand_config_list[1].weight_bits, 8)

    def test_config_set_api(self):
        # *Note: this test is only for improving the code coverage and can be removed once the test_common is enabled.
        from neural_compressor.common.base_config import config_registry, get_all_config_set_from_config_registry
        from neural_compressor.onnxrt.quantization.config import FRAMEWORK_NAME

        config_set = get_all_config_set_from_config_registry(fwk_name=FRAMEWORK_NAME)
        self.assertEqual(len(config_set), len(config_registry.registered_configs[FRAMEWORK_NAME]))


if __name__ == "__main__":
    unittest.main()
