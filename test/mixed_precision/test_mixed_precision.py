#
#  -*- coding: utf-8 -*-
#
import os
import shutil
import unittest

import numpy as np
import tensorflow as tf
from onnx import TensorProto, helper
from packaging.version import Version
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

import neural_compressor.adaptor.pytorch as nc_torch
from neural_compressor import mix_precision
from neural_compressor.adaptor.torch_utils.bf16_convert import BF16ModuleWrapper
from neural_compressor.config import MixedPrecisionConfig, TuningCriterion
from neural_compressor.mix_precision import fit
from neural_compressor.utils.utility import CpuInfo, LazyImport

PT_VERSION = nc_torch.get_torch_version()


def build_matmul_model():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 1, 5, 1])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 1, 5, 1])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 1, 5, 1])
    H = helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 1, 5, 1])

    matmul_node = helper.make_node("MatMul", ["A", "B"], ["C"], name="Matmul")
    e_value = np.random.randint(2, size=(5)).astype(np.float32)
    E_init = helper.make_tensor("E", TensorProto.FLOAT, [1, 1, 5, 1], e_value.reshape(5).tolist())
    add = helper.make_node("Add", ["C", "E"], ["D"], name="add")

    f_value = np.random.randint(2, size=(5)).astype(np.float32)
    F_init = helper.make_tensor("F", TensorProto.FLOAT, [1, 1, 5, 1], e_value.reshape(5).tolist())
    add2 = helper.make_node("Add", ["D", "F"], ["H"], name="add2")

    graph = helper.make_graph([matmul_node, add, add2], "test_graph_1", [A, B], [H], [E_init, F_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 16)]})
    return model


def build_tf_graph():
    input_node = node_def_pb2.NodeDef()
    input_node.name = "input"
    input_node.op = "Placeholder"
    input_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))

    conv1_weight_node = node_def_pb2.NodeDef()
    conv1_weight_node.name = "conv1_weights"
    conv1_weight_node.op = "Const"
    conv1_weight_value = np.float32(np.abs(np.random.randn(3, 3, 3, 32)))
    conv1_weight_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv1_weight_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
                conv1_weight_value, conv1_weight_value.dtype.type, conv1_weight_value.shape
            )
        )
    )

    conv1_node = node_def_pb2.NodeDef()
    conv1_node.name = "conv1"
    conv1_node.op = "Conv2D"
    conv1_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv1_node.input.extend([input_node.name, conv1_weight_node.name])
    conv1_node.attr["strides"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv1_node.attr["dilations"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv1_node.attr["padding"].CopyFrom(attr_value_pb2.AttrValue(s=b"SAME"))
    conv1_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    bias_node = node_def_pb2.NodeDef()
    bias_node.name = "conv1_bias"
    bias_node.op = "Const"
    bias_value = np.float32(np.abs(np.random.randn(32)))
    bias_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(bias_value, bias_value.dtype.type, bias_value.shape)
        )
    )

    bias_add_node = node_def_pb2.NodeDef()
    bias_add_node.name = "conv1_bias_add"
    bias_add_node.op = "BiasAdd"
    bias_add_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_add_node.input.extend([conv1_node.name, bias_node.name])
    bias_add_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    relu_node = node_def_pb2.NodeDef()
    relu_node.op = "Relu"
    relu_node.name = "relu"
    relu_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    relu_node.input.extend([bias_add_node.name])

    conv2_weight_node = node_def_pb2.NodeDef()
    conv2_weight_node.name = "conv2_weights"
    conv2_weight_node.op = "Const"
    conv2_weight_value = np.float32(np.abs(np.random.randn(3, 3, 32, 32)))
    conv2_weight_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv2_weight_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
                conv2_weight_value, conv2_weight_value.dtype.type, conv2_weight_value.shape
            )
        )
    )

    conv2_node = node_def_pb2.NodeDef()
    conv2_node.name = "conv2"
    conv2_node.op = "Conv2D"
    conv2_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv2_node.input.extend([relu_node.name, conv2_weight_node.name])
    conv2_node.attr["strides"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv2_node.attr["dilations"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv2_node.attr["padding"].CopyFrom(attr_value_pb2.AttrValue(s=b"SAME"))
    conv2_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    bias_node2 = node_def_pb2.NodeDef()
    bias_node2.name = "conv2_bias"
    bias_node2.op = "Const"
    bias_value2 = np.float32(np.abs(np.random.randn(32)))
    bias_node2.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_node2.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(bias_value2, bias_value2.dtype.type, bias_value2.shape)
        )
    )

    bias_add_node2 = node_def_pb2.NodeDef()
    bias_add_node2.name = "conv2_bias_add"
    bias_add_node2.op = "BiasAdd"
    bias_add_node2.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_add_node2.input.extend([conv2_node.name, bias_node2.name])
    bias_add_node2.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    relu_node2 = node_def_pb2.NodeDef()
    relu_node2.op = "Relu"
    relu_node2.name = "relu2"
    relu_node2.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    relu_node2.input.extend([bias_add_node2.name])

    conv3_weight_node = node_def_pb2.NodeDef()
    conv3_weight_node.name = "conv3_weights"
    conv3_weight_node.op = "Const"
    conv3_weight_value = np.float32(np.abs(np.random.randn(3, 3, 32, 32)))
    conv3_weight_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv3_weight_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
                conv3_weight_value, conv3_weight_value.dtype.type, conv3_weight_value.shape
            )
        )
    )

    conv3_node = node_def_pb2.NodeDef()
    conv3_node.name = "conv3"
    conv3_node.op = "Conv2D"
    conv3_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv3_node.input.extend([relu_node2.name, conv3_weight_node.name])
    conv3_node.attr["strides"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv3_node.attr["dilations"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv3_node.attr["padding"].CopyFrom(attr_value_pb2.AttrValue(s=b"SAME"))
    conv3_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    identity_node = node_def_pb2.NodeDef()
    identity_node.name = "final"
    identity_node.op = "Identity"
    identity_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    identity_node.input.extend([conv3_node.name])

    test_graph = graph_pb2.GraphDef()

    test_graph.node.extend(
        [
            input_node,
            conv1_weight_node,
            conv1_node,
            bias_node,
            bias_add_node,
            # cast_node,
            relu_node,
            # cast2_node,
            conv2_weight_node,
            conv2_node,
            bias_node2,
            bias_add_node2,
            relu_node2,
            conv3_weight_node,
            conv3_node,
            identity_node,
        ]
    )
    return test_graph


def build_pt_model():
    resnet18 = LazyImport("torchvision.models.resnet18")
    return resnet18()


def build_yaml():
    fake_yaml = """
        device: gpu
        model:
          name: test
          framework: onnxrt_qlinearops

        mixed_precision:
          precisions: fp16

        evaluation:
          accuracy:
            metric:
              MSE:
                compare_label: False
            dataloader:
              dataset:
                dummy:
                  shape: [[1,1,5,5], [1,1,5,1]]
                  label: True
        """
    with open("test.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)


class MatmulDataloader:
    def __init__(self):
        self.batch_size = 1
        self.data = []
        self.label = []
        for i in range(3):
            self.data.append(
                [np.random.randn(1, 1, 5, 5).astype("float32"), np.random.randn(1, 1, 5, 1).astype("float32")]
            )
            self.label.append(np.random.randn(1, 1, 5, 1).astype("float32"))

    def __iter__(self):
        for data, label in zip(self.data, self.label):
            yield data, label


class Metric:
    def update(self, preds, labels):
        pass

    def reset(self):
        pass

    def result(self):
        return 0.5


class TestMixedPrecisionOnNonEnabledHost(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.onnx_model = build_matmul_model()
        self.tf_model = build_tf_graph()

    @unittest.skipIf(CpuInfo().bf16, "skip since hardware support bf16")
    def test_on_non_enabled_host_tf(self):
        conf = MixedPrecisionConfig()
        with self.assertRaises(SystemExit) as cm:
            output_model = mix_precision.fit(self.tf_model, conf)
        self.assertEqual(cm.exception.code, 0)

    def test_on_non_enabled_dtype(self):
        # test onnx
        conf = MixedPrecisionConfig()
        with self.assertRaises(SystemExit) as cm:
            output_model = mix_precision.fit(self.onnx_model, conf)
        self.assertEqual(cm.exception.code, 0)

        conf = MixedPrecisionConfig(precisions="fp16")
        with self.assertRaises(SystemExit) as cm:
            output_model = mix_precision.fit(self.tf_model, conf)
        self.assertEqual(cm.exception.code, 0)


class TestMixedPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.environ["FORCE_FP16"] = "1"
        os.environ["FORCE_BF16"] = "1"
        self.onnx_model = build_matmul_model()
        self.matmul_dataloader = MatmulDataloader()
        self.tf_model = build_tf_graph()
        self.pt_model = build_pt_model()
        build_yaml()

    @classmethod
    def tearDownClass(self):
        del os.environ["FORCE_FP16"]
        del os.environ["FORCE_BF16"]
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("./nc_workspace", ignore_errors=True)
        os.remove("test.yaml")

    def test_mixed_precision_with_evaluation(self):
        from neural_compressor.data import DataLoader
        from neural_compressor.metric.metric import ONNXRT_QL_METRICS

        # test onnx
        conf = MixedPrecisionConfig(device="gpu", backend="onnxrt_cuda_ep")

        # output_model = mix_precision.fit(self.onnx_model, conf)
        # self.assertTrue(any([i.op_type == 'Cast' for i in output_model.nodes()]))

        tuning_criterion = TuningCriterion(max_trials=3, timeout=1000000)
        conf = MixedPrecisionConfig(
            device="gpu", tuning_criterion=tuning_criterion, backend="onnxrt_cuda_ep", precisions="fp16"
        )
        output_model = mix_precision.fit(
            self.onnx_model, conf, eval_dataloader=self.matmul_dataloader, eval_metric=ONNXRT_QL_METRICS["MSE"]()
        )
        self.assertTrue(any([i.op_type == "Cast" for i in output_model.nodes()]))

    def test_mixed_precision_with_evaluation_old_api(self):
        from neural_compressor.conf.config import MixedPrecision_Conf
        from neural_compressor.experimental import MixedPrecision

        converter = MixedPrecision(MixedPrecision_Conf("test.yaml"))
        converter.model = self.onnx_model
        output_model = converter.fit()
        self.assertTrue(any([i.op_type != "Cast" for i in output_model.nodes()]))

    def test_mixed_precision_with_eval_func(self):
        def eval(model):
            return 0.5

        result = [0.0, 0.1, 0.102, 0.1003, 0.1005, 0.1004, 0.1002]
        perf = [0.1, 0.5, 0.6, 0.7, 0.5, 0.4, 0.5]
        import time

        def eval2(model):
            del perf[0]
            del result[0]
            time.sleep(perf[0])
            return result[0]

        conf = MixedPrecisionConfig(
            inputs="input",
            outputs="final",
        )

        output_model = mix_precision.fit(
            self.tf_model,
            conf,
            eval_func=eval,
        )
        self.assertTrue(any([i.op == "Cast" for i in output_model.graph_def.node]))
        self.assertEqual(conf.inputs, "input")
        self.assertEqual(conf.outputs, "final")

        tuning_criterion = TuningCriterion(max_trials=4, timeout=500)
        conf = MixedPrecisionConfig(tuning_criterion=tuning_criterion)
        output_model = mix_precision.fit(
            self.tf_model,
            conf,
            eval_func=eval2,
        )
        self.assertTrue(any([i.op == "Cast" for i in output_model.graph_def.node]))

        tuning_criterion = TuningCriterion(max_trials=1, timeout=100)
        conf = MixedPrecisionConfig(inputs="input", outputs="final, test", tuning_criterion=tuning_criterion)
        output_model = mix_precision.fit(
            self.tf_model,
            conf,
            eval_func=eval,
        )
        self.assertTrue(any([i.op == "Cast" for i in output_model.graph_def.node]))

        output_model = fit(self.tf_model, conf, eval)
        self.assertTrue(any([i.op == "Cast" for i in output_model.graph_def.node]))

    def test_mixed_precision_with_quant_level_1(self):
        result = [0.0, 0.1, 0.102]

        def eval_func(model):
            del result[0]
            return result[0]

        conf = MixedPrecisionConfig(inputs="input", outputs="final", quant_level="auto")

        output_model = mix_precision.fit(self.tf_model, conf, eval_func=eval_func)
        self.assertTrue(any([i.op == "Cast" for i in output_model.graph_def.node]))
        self.assertEqual(conf.inputs, "input")
        self.assertEqual(conf.outputs, "final")

    def test_mixed_precision_with_quant_level_2(self):
        result = [0.0, 1, 0.9, 1.1]

        # meet acc if fallback all conv
        def eval_func(model):
            del result[0]
            return result[0]

        conf = MixedPrecisionConfig(inputs="input", outputs="final", quant_level="auto")

        output_model = mix_precision.fit(self.tf_model, conf, eval_func=eval_func)
        # no cast in output model
        self.assertFalse(any([i.op == "Cast" for i in output_model.graph_def.node]))

    def test_mixed_precision_with_quant_level_3(self):
        result = [0.0, 1, 0.9, 0.9, 1.1]

        # meet acc if fallback 1 conv
        def eval_func(model):
            del result[0]
            return result[0]

        conf = MixedPrecisionConfig(inputs="input", outputs="final", quant_level="auto")

        output_model = mix_precision.fit(self.tf_model, conf, eval_func=eval_func)
        # no cast in output model
        count_cast = 0
        for node in output_model.graph_def.node:
            if node.op == "Cast":
                count_cast += 1
        self.assertEqual(count_cast, 4)

    def test_mixed_precision_with_quant_level_4(self):
        result = [0.0, 1, 0.9, 0.9, 1.1]

        # meet acc if fallback the second conv
        def eval_func(model):
            del result[0]
            return result[0]

        conf = MixedPrecisionConfig(inputs="input", outputs="final", quant_level=1)

        output_model = mix_precision.fit(self.tf_model, conf, eval_func=eval_func)
        # no cast in output model
        count_cast = 0
        for node in output_model.graph_def.node:
            if node.op == "Cast":
                count_cast += 1
        self.assertEqual(count_cast, 4)

    def test_mixed_precision_with_quant_level_5(self):
        result = [0.0, 1, 0.9, 0.9, 0.9]

        # meet not meet
        def eval_func(model):
            del result[0]
            return result[0]

        conf = MixedPrecisionConfig(inputs="input", outputs="final", quant_level=0)

        output_model = mix_precision.fit(self.tf_model, conf, eval_func=eval_func)
        self.assertIsNone(output_model)

    @unittest.skipIf(
        PT_VERSION.release < Version("1.11.0").release, "Please use PyTroch 1.11 or higher version for mixed precision."
    )
    def test_mixed_precision_with_eval_func_pt(self):
        torch = LazyImport("torch")

        def eval(model):
            return 0.5

        conf = MixedPrecisionConfig()
        output_model = mix_precision.fit(
            self.pt_model,
            conf,
            eval_func=eval,
        )
        self.assertTrue(isinstance(output_model.model.fc, BF16ModuleWrapper))
        op_name_dict = {
            "fc": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            }
        }
        conf = MixedPrecisionConfig(op_name_dict=op_name_dict)
        output_model = mix_precision.fit(
            self.pt_model,
            conf,
            eval_func=eval,
        )
        self.assertTrue(isinstance(output_model.model.fc.weight.dtype, type(torch.float32)))
        op_type_dict = {
            "Linear": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            }
        }
        conf = MixedPrecisionConfig(op_type_dict=op_type_dict)
        output_model = mix_precision.fit(
            self.pt_model,
            conf,
            eval_func=eval,
        )
        self.assertTrue(isinstance(output_model.model.fc.weight.dtype, type(torch.float32)))


if __name__ == "__main__":
    unittest.main()
