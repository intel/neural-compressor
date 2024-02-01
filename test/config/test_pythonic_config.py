"""Tests for pythonic config file."""

import copy
import os
import shutil
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import torch
from onnx import TensorProto, helper
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util
from torch import nn

from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.adaptor.torch_utils.bf16_convert import BF16ModuleWrapper
from neural_compressor.conf.pythonic_config import ActivationConf, OpQuantConf, WeightConf, config
from neural_compressor.data import Datasets
from neural_compressor.experimental import NAS, Distillation, Quantization, common
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.experimental.pruning_v2 import Pruning


def build_matmul_model():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
    B_init = helper.make_tensor(
        "B", TensorProto.FLOAT, [1, 1, 5, 1], np.random.random([1, 1, 5, 1]).reshape(5).tolist()
    )
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 1, 5, 1])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 1, 5, 1])
    H = helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 1, 5, 1])

    matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"], name="Matmul")
    e_value = np.random.randint(2, size=(5)).astype(np.float32)
    E_init = helper.make_tensor("E", TensorProto.FLOAT, [1, 1, 5, 1], e_value.reshape(5).tolist())
    add = onnx.helper.make_node("Add", ["C", "E"], ["D"], name="add")
    f_value = np.random.randint(2, size=(5)).astype(np.float32)
    F_init = helper.make_tensor("F", TensorProto.FLOAT, [1, 1, 5, 1], e_value.reshape(5).tolist())
    add2 = onnx.helper.make_node("Add", ["D", "F"], ["H"], name="add2")
    graph = helper.make_graph([matmul_node, add, add2], "test_graph_1", [A], [H], [E_init, F_init, B_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    return model


def build_conv2d_model():
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
    bias_add_node.name = "out"
    bias_add_node.op = "BiasAdd"
    bias_add_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_add_node.input.extend([conv1_node.name, bias_node.name])
    bias_add_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    test_graph = graph_pb2.GraphDef()
    test_graph.node.extend(
        [
            input_node,
            conv1_weight_node,
            conv1_node,
            bias_node,
            bias_add_node,
        ]
    )
    return test_graph


class ConvNet(torch.nn.Module):
    def __init__(self, channels, dimensions):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, channels, (3, 3), padding=1)
        self.avg_pooling = torch.nn.AvgPool2d((64, 64))
        self.dense = torch.nn.Linear(channels, dimensions)
        self.out = torch.nn.Linear(dimensions, 2)
        self.activation = torch.nn.Softmax()

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.avg_pooling(outputs).squeeze()
        outputs = self.dense(outputs)
        outputs = self.out(outputs)
        outputs = self.activation(outputs)
        return outputs


def model_builder(model_arch_params):
    channels = model_arch_params["channels"]
    dimensions = model_arch_params["dimensions"]
    return ConvNet(channels, dimensions)


class torch_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.linear = nn.Linear(224 * 224, 5)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(1, -1)
        x = self.linear(x)
        return x


class TestPythonicConf(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./nc_workspace", ignore_errors=True)

    def test_config_setting(self):
        config.quantization.inputs = ["image"]
        config.quantization.outputs = ["out"]
        config.quantization.approach = "post_training_dynamic_quant"
        config.quantization.device = "gpu"
        config.quantization.op_type_dict = {"Conv": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}}
        config.quantization.op_name_dict = {
            "layer1.0.conv1": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}}
        }
        config.quantization.strategy = "mse"
        config.quantization.objective = "accuracy"
        config.quantization.timeout = 100
        config.quantization.max_trials = 100
        config.quantization.accuracy_criterion.relative = 0.5
        config.quantization.reduce_range = False
        config.quantization.use_bf16 = False
        config.benchmark.cores_per_instance = 10

        self.assertEqual(config.quantization.inputs, ["image"])
        self.assertEqual(config.quantization.outputs, ["out"])
        self.assertEqual(config.quantization.approach, "post_training_dynamic_quant")
        self.assertEqual(config.quantization.device, "gpu")
        self.assertEqual(
            config.quantization.op_type_dict,
            {"Conv": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}},
        )
        self.assertEqual(
            config.quantization.op_name_dict,
            {"layer1.0.conv1": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}}},
        )
        self.assertEqual(config.quantization.strategy, "mse")
        self.assertEqual(config.quantization.objective, "accuracy")
        self.assertEqual(config.quantization.timeout, 100)
        self.assertEqual(config.quantization.max_trials, 100)
        self.assertEqual(config.quantization.accuracy_criterion.relative, 0.5)
        self.assertEqual(config.benchmark.cores_per_instance, 10)

        config.quantization.accuracy_criterion.absolute = 0.4
        self.assertEqual(config.quantization.accuracy_criterion.absolute, 0.4)
        self.assertEqual(config.quantization.accuracy_criterion.relative, None)

        config.onnxruntime.precisions = ["int8", "uint8"]
        config.onnxruntime.graph_optimization_level = "DISABLE_ALL"
        q = Quantization(config)
        q.model = build_matmul_model()
        self.assertEqual(q.conf.usr_cfg.reduce_range, False)
        self.assertEqual(q.conf.usr_cfg.use_bf16, False)
        q.pre_process()
        self.assertEqual(q.strategy.adaptor.query_handler.get_precisions(), ["int8", "uint8"])
        self.assertNotEqual(config.mxnet, None)
        self.assertNotEqual(config.tensorflow, None)
        self.assertNotEqual(config.pytorch, None)
        self.assertNotEqual(config.keras, None)

    def test_weight_activation_op(self):
        opconf = OpQuantConf()
        self.assertEqual(opconf.op_type, None)

        opconf = OpQuantConf("MatMul")
        self.assertEqual(opconf.op_type, "MatMul")
        self.assertNotEqual(opconf.weight, None)
        self.assertNotEqual(opconf.activation, None)

        opconf.weight.datatype = ["int8"]
        opconf.activation.datatype = ["uint8"]
        opconf.weight.scheme = ["asym"]
        opconf.activation.scheme = ["sym"]
        opconf.weight.granularity = ["per_channel"]
        opconf.activation.granularity = ["per_tensor"]
        opconf.weight.algorithm = ["minmax"]
        opconf.activation.algorithm = ["minmax"]
        self.assertEqual(opconf.weight.datatype, ["int8"])
        self.assertEqual(opconf.activation.datatype, ["uint8"])
        self.assertEqual(opconf.weight.scheme, ["asym"])
        self.assertEqual(opconf.activation.scheme, ["sym"])
        self.assertEqual(opconf.weight.granularity, ["per_channel"])
        self.assertEqual(opconf.activation.granularity, ["per_tensor"])
        self.assertEqual(opconf.weight.algorithm, ["minmax"])
        self.assertEqual(opconf.activation.algorithm, ["minmax"])

    def test_quantization(self):
        q = Quantization(config)
        q.model = build_matmul_model()
        q_model = q()
        self.assertTrue(any([i.name.endswith("_quant") for i in q_model.nodes()]))

        config.onnxruntime.precisions = ["fp32"]
        q = Quantization(config)
        q.model = build_matmul_model()
        q_model = q()
        self.assertTrue(all([not i.name.endswith("_quant") for i in q_model.nodes()]))

    def test_distillation(self):
        config.quantization.device = "cpu"
        distiller = Distillation(config)
        model = ConvNet(16, 32)
        origin_weight = copy.deepcopy(model.out.weight)
        distiller.model = model
        distiller.teacher_model = ConvNet(16, 32)

        # Customized train, evaluation
        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(32, 3, 64, 64), low=0.0, high=1.0, label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        def train_func(model):
            epochs = 3
            iters = 10
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                for image, target in dummy_dataloader:
                    print(".", end="")
                    cnt += 1
                    output = model(image).unsqueeze(dim=0)
                    loss = criterion(output, target)
                    loss = distiller.on_after_compute_loss(image, output, loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if cnt >= iters:
                        break

        def eval_func(model):
            model.eval()
            acc = 0
            for image, target in dummy_dataloader:
                output = model(image).cpu().detach().numpy()
                acc += np.sum(output == target)
            return {"acc": acc / len(dummy_dataset)}

        distiller.train_func = train_func
        distiller.eval_func = eval_func
        model = distiller()
        weight = model.model.out.weight
        self.assertTrue(torch.any(weight != origin_weight))

    def test_pruning(self):
        prune = Pruning(config)
        model = ConvNet(16, 32)
        origin_weight = copy.deepcopy(model.out.weight)
        prune.model = model

        # Customized train, evaluation
        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(32, 3, 64, 64), low=0.0, high=1.0, label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        def train_func(model):
            epochs = 3
            iters = 10
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                prune.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    print(".", end="")
                    cnt += 1
                    prune.on_step_begin(cnt)
                    output = model(image).unsqueeze(dim=0)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    prune.on_step_end()
                    if cnt >= iters:
                        break
                prune.on_epoch_end()

        def eval_func(model):
            model.eval()
            acc = 0
            for image, target in dummy_dataloader:
                output = model(image).cpu().detach().numpy()
                acc += np.sum(output == target)
            return {"acc": acc / len(dummy_dataset)}

        prune.train_func = train_func
        prune.eval_func = eval_func
        model = prune()
        weight = model.model.out.weight
        self.assertTrue(torch.any(weight != origin_weight))

    def test_use_bf16(self):
        config.quantization.device = "cpu"
        config.quantization.approach = "post_training_dynamic_quant"
        config.quantization.use_bf16 = False
        q = Quantization(config)
        q.model = torch_model()
        os.environ["FORCE_BF16"] = "1"
        q_model = q()
        del os.environ["FORCE_BF16"]
        self.assertEqual(isinstance(q_model.model.linear, BF16ModuleWrapper), False)

    def test_quantization_pytorch(self):
        config.quantization.device = "cpu"
        config.quantization.backend = "default"
        config.quantization.approach = "post_training_dynamic_quant"
        config.quantization.use_bf16 = False
        q = Quantization(config)
        q.model = torch_model()
        q_model = q()
        self.assertEqual(isinstance(q_model.model.linear, torch.nn.quantized.dynamic.modules.linear.Linear), True)


class TestTFPyhonicConf(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./nc_workspace", ignore_errors=True)

    def test_tf_quantization(self):
        config.quantization.inputs = ["input"]
        config.quantization.outputs = ["out"]
        config.quantization.approach = "post_training_static_quant"
        config.quantization.device = "cpu"
        config.quantization.strategy = "basic"
        config.quantization.objective = "accuracy"
        config.quantization.timeout = 100
        config.quantization.accuracy_criterion.relative = 0.5
        config.quantization.reduce_range = False

        q = Quantization(config)
        q.model = build_conv2d_model()
        dataset = q.dataset("dummy", shape=(1, 224, 224, 3), label=True)
        q.calib_dataloader = common.DataLoader(dataset)
        q_model = q()

        self.assertTrue(any([i.name.endswith("_requantize") for i in q_model.graph_def.node]))


if __name__ == "__main__":
    unittest.main()
