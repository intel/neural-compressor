#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import yaml
import numpy as np
import tensorflow as tf
from onnx import helper, TensorProto, numpy_helper
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from neural_compressor.utils.utility import LazyImport, CpuInfo
from neural_compressor.adaptor.pytorch import PyTorchVersionMode
import neural_compressor.adaptor.pytorch as nc_torch
from neural_compressor.adaptor.torch_utils.bf16_convert import BF16ModuleWrapper
import shutil

PT_VERSION = nc_torch.get_torch_version()
def build_matmul_model():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 5, 1])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 1])
    D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 1])
    H = helper.make_tensor_value_info('H', TensorProto.FLOAT, [1, 1, 5, 1])
 
    matmul_node = helper.make_node('MatMul', ['A', 'B'], ['C'], name='Matmul')
    e_value = np.random.randint(2, size=(5)).astype(np.float32)
    E_init = helper.make_tensor('E', TensorProto.FLOAT, [1, 1, 5, 1], e_value.reshape(5).tolist())
    add = helper.make_node('Add', ['C', 'E'], ['D'], name='add')
 
    f_value = np.random.randint(2, size=(5)).astype(np.float32)
    F_init = helper.make_tensor('F', TensorProto.FLOAT, [1, 1, 5, 1], e_value.reshape(5).tolist())
    add2 = helper.make_node('Add', ['D', 'F'], ['H'], name='add2')
 
    graph = helper.make_graph([matmul_node, add, add2], 'test_graph_1', [A, B], [H], [E_init, F_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
    return model

def build_tf_graph():
    input_node = node_def_pb2.NodeDef()
    input_node.name = "input"
    input_node.op = "Placeholder"
    input_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))

    conv1_weight_node = node_def_pb2.NodeDef()
    conv1_weight_node.name = "conv1_weights"
    conv1_weight_node.op = "Const"
    conv1_weight_value = np.float32(np.abs(np.random.randn(3,3,3,32)))
    conv1_weight_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv1_weight_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
        tensor=tensor_util.make_tensor_proto(
            conv1_weight_value, conv1_weight_value.dtype.type, conv1_weight_value.shape)))

    conv1_node = node_def_pb2.NodeDef()
    conv1_node.name = "conv1"
    conv1_node.op = "Conv2D"
    conv1_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    conv1_node.input.extend([input_node.name, conv1_weight_node.name])
    conv1_node.attr['strides'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv1_node.attr['dilations'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv1_node.attr['padding'].CopyFrom(attr_value_pb2.AttrValue(s=b'SAME'))
    conv1_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))

    bias_node = node_def_pb2.NodeDef()
    bias_node.name = "conv1_bias"
    bias_node.op = "Const"
    bias_value = np.float32(np.abs(np.random.randn(32)))
    bias_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
        bias_value, bias_value.dtype.type, bias_value.shape)))

    bias_add_node = node_def_pb2.NodeDef()
    bias_add_node.name = "conv1_bias_add"
    bias_add_node.op = "BiasAdd"
    bias_add_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_add_node.input.extend([conv1_node.name, bias_node.name])
    bias_add_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))

    relu_node = node_def_pb2.NodeDef()
    relu_node.op = "Relu"
    relu_node.name = "relu"
    relu_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    relu_node.input.extend([bias_add_node.name])

    conv2_weight_node = node_def_pb2.NodeDef()
    conv2_weight_node.name = "conv2_weights"
    conv2_weight_node.op = "Const"
    conv2_weight_value = np.float32(np.abs(np.random.randn(3,3,32,32)))
    conv2_weight_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv2_weight_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
        tensor=tensor_util.make_tensor_proto(
            conv2_weight_value, conv2_weight_value.dtype.type, conv2_weight_value.shape)))

    conv2_node = node_def_pb2.NodeDef()
    conv2_node.name = "conv2"
    conv2_node.op = "Conv2D"
    conv2_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    conv2_node.input.extend([relu_node.name, conv2_weight_node.name])
    conv2_node.attr['strides'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv2_node.attr['dilations'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv2_node.attr['padding'].CopyFrom(attr_value_pb2.AttrValue(s=b'SAME'))
    conv2_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))

    bias_node2 = node_def_pb2.NodeDef()
    bias_node2.name = "conv2_bias"
    bias_node2.op = "Const"
    bias_value2 = np.float32(np.abs(np.random.randn(32)))
    bias_node2.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_node2.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
        bias_value2, bias_value2.dtype.type, bias_value2.shape)))

    bias_add_node2 = node_def_pb2.NodeDef()
    bias_add_node2.name = "conv2_bias_add"
    bias_add_node2.op = "BiasAdd"
    bias_add_node2.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_add_node2.input.extend([conv2_node.name, bias_node2.name])
    bias_add_node2.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))

    relu_node2 = node_def_pb2.NodeDef()
    relu_node2.op = "Relu"
    relu_node2.name = "relu2"
    relu_node2.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    relu_node2.input.extend([bias_add_node2.name])

    conv3_weight_node = node_def_pb2.NodeDef()
    conv3_weight_node.name = "conv3_weights"
    conv3_weight_node.op = "Const"
    conv3_weight_value = np.float32(np.abs(np.random.randn(3,3,32,32)))
    conv3_weight_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv3_weight_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
        tensor=tensor_util.make_tensor_proto(
            conv3_weight_value, conv3_weight_value.dtype.type, conv3_weight_value.shape)))

    conv3_node = node_def_pb2.NodeDef()
    conv3_node.name = "conv3"
    conv3_node.op = "Conv2D"
    conv3_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    conv3_node.input.extend([relu_node2.name, conv3_weight_node.name])
    conv3_node.attr['strides'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv3_node.attr['dilations'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv3_node.attr['padding'].CopyFrom(attr_value_pb2.AttrValue(s=b'SAME'))
    conv3_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))

    identity_node = node_def_pb2.NodeDef()
    identity_node.name = "final"
    identity_node.op = "Identity"
    identity_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    identity_node.input.extend([conv3_node.name])

    test_graph = graph_pb2.GraphDef()

    test_graph.node.extend([input_node,
                            conv1_weight_node,
                            conv1_node,
                            bias_node,
                            bias_add_node,
                            #cast_node,
                            relu_node,
                            #cast2_node,
                            conv2_weight_node,
                            conv2_node,
                            bias_node2,
                            bias_add_node2,
                            relu_node2,
                            conv3_weight_node,
                            conv3_node,
                            identity_node
                            ])
    return test_graph

def build_pt_model():
    resnet18 = LazyImport("torchvision.models.resnet18")
    return resnet18()

def build_yaml():
    fake_yaml = """
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
                  shape: [[5,1,5,5], [5,1,5,1]]
                  label: True
        """
    with open("test.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)

class MatmulDataset:
    def __init__(self):
        self.data = []
        self.label = []
        for i in range(3):
            self.data.append([np.random.randn(1,5,5).astype('float32'), 
                              np.random.randn(1,5,1).astype('float32')])
            self.label.append(np.random.randn(1,5,1).astype('float32'))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

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

    def test_on_non_enabled_host(self):
        from neural_compressor.experimental import MixedPrecision
        from neural_compressor import conf

        # test onnx
        conf.model.framework = 'onnxrt_qlinearops'
        converter = MixedPrecision(conf)
        converter.model = self.onnx_model
        converter.precisions = 'fp16'
        with self.assertRaises(SystemExit) as cm:
            output_model = converter.fit()
        self.assertEqual(cm.exception.code, 0)

    @unittest.skipIf(CpuInfo().bf16, 'skip since hardware support bf16')
    def test_on_non_enabled_host_tf(self):
        from neural_compressor.experimental import MixedPrecision
        from neural_compressor import conf
        conf.model.framework = 'tensorflow'
        converter = MixedPrecision(conf)
        converter.model = self.tf_model
        converter.precisions = 'bf16'
        with self.assertRaises(SystemExit) as cm:
            output_model = converter.fit()
        self.assertEqual(cm.exception.code, 0)

    def test_on_non_enabled_dtype(self):
        from neural_compressor.experimental import MixedPrecision
        from neural_compressor import conf

        # test onnx
        conf.model.framework = 'onnxrt_qlinearops'
        converter = MixedPrecision(conf)
        converter.model = self.onnx_model
        converter.precisions = 'bf16'
        with self.assertRaises(SystemExit) as cm:
            output_model = converter.fit()
        self.assertEqual(cm.exception.code, 0)

        conf.model.framework = 'tensorflow'
        converter = MixedPrecision(conf)
        converter.model = self.tf_model
        converter.precisions = 'fp16'
        with self.assertRaises(SystemExit) as cm:
            output_model = converter.fit()
        self.assertEqual(cm.exception.code, 0)

class TestMixedPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.environ['FORCE_FP16'] = '1'
        os.environ['FORCE_BF16'] = '1'
        self.onnx_model = build_matmul_model()
        self.matmul_dataset = MatmulDataset()
        self.tf_model = build_tf_graph()
        self.pt_model = build_pt_model()
        build_yaml()

    @classmethod
    def tearDownClass(self):
        del os.environ['FORCE_FP16']
        del os.environ['FORCE_BF16']
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("./nc_workspace", ignore_errors=True)
        os.remove("test.yaml")

    def test_mixed_precision_with_evaluation(self):
        from neural_compressor.experimental import MixedPrecision, common
        from neural_compressor import conf
        # test onnx
        conf.model.framework = 'onnxrt_qlinearops'
        conf.mixed_precision.precisions = 'fp16'
        conf.tuning.workspace.path = './saved'
        converter = MixedPrecision(conf)
        self.assertEqual(str(converter), 'MixedPrecision')
        converter.model = self.onnx_model
        output_model = converter.fit()
        self.assertFalse(any([i.op_type == 'Cast' for i in output_model.nodes()]))

        conf.tuning.exit_policy.max_trials = 3
        conf.tuning.exit_policy.timeout = 50
        conf.evaluation.accuracy.metric = {'MSE': {'compare_label': False}}
        converter = MixedPrecision(conf)
        converter.eval_dataloader = common.DataLoader(self.matmul_dataset)
        converter.model = self.onnx_model
        output_model = converter.fit()
        self.assertFalse(any([i.op_type == 'Cast' for i in output_model.nodes()]))

        from neural_compressor.conf.config import MixedPrecision_Conf
        converter = MixedPrecision(MixedPrecision_Conf('test.yaml'))
        converter.model = self.onnx_model
        output_model = converter.fit()
        self.assertFalse(any([i.op_type == 'Cast' for i in output_model.nodes()]))

    def test_mixed_precision_with_eval_func(self):
        def eval(model):
            return 0.5

        result = [0., 0.1, 0.102, 0.1006, 0.1005, 0.1004, 0.1002]
        def eval2(model):
            del result[0]
            return result[0]
 
        from neural_compressor.experimental import MixedPrecision, common
        from neural_compressor import conf
        conf.model.framework = 'onnxrt_qlinearops'
        converter = MixedPrecision(conf)
        converter.precisions = 'fp16'
        converter.model = self.onnx_model
        converter.eval_dataloader = common.DataLoader(self.matmul_dataset)
        converter.metric = Metric()
        output_model = converter.fit()
        self.assertFalse(any([i.op_type == 'Cast' for i in output_model.nodes()]))
        converter.metric = common.Metric(Metric)
        output_model = converter.fit()
        self.assertFalse(any([i.op_type == 'Cast' for i in output_model.nodes()]))

        converter = MixedPrecision()
        converter.precisions = ['bf16', 'fp32']
        converter.input = 'input'
        converter.output = 'final'
        converter.model = self.tf_model
        converter.eval_func = eval
        output_model = converter.fit()
        self.assertTrue(any([i.op == 'Cast' for i in output_model.graph_def.node]))
        self.assertEqual(converter.precisions, ['bf16', 'fp32'])
        self.assertEqual(converter.input, 'input')
        self.assertEqual(converter.output, 'final')

        conf.tuning.exit_policy.max_trials = 10
        conf.tuning.exit_policy.timeout = 500
        conf.model.framework = 'tensorflow'
        converter = MixedPrecision(conf)
        converter.precisions = 'bf16'
        converter.model = common.Model(self.tf_model)
        converter.eval_func = eval2
        output_model = converter.fit()
        self.assertTrue(any([i.op == 'Cast' for i in output_model.graph_def.node]))

        conf.tuning.exit_policy.max_trials = 1
        conf.tuning.exit_policy.timeout = 100
        converter = MixedPrecision()
        converter.precisions = ['bf16', 'fp32']
        converter.input = 'input'
        converter.output = 'final, test'
        converter.model = self.tf_model
        converter.eval_func = eval
        output_model = converter.fit()
        self.assertTrue(any([i.op == 'Cast' for i in output_model.graph_def.node]))
 
    @unittest.skipIf(PT_VERSION < PyTorchVersionMode.PT111.value,
      "Please use PyTroch 1.11 or higher version for mixed precision.")
    def test_mixed_precision_with_eval_func_pt(self):
        def eval(model):
            return 0.5
        from neural_compressor.experimental import MixedPrecision
        from neural_compressor import conf
        conf.model.framework = "pytorch"
        converter = MixedPrecision(conf)
        converter.precisions = 'bf16'
        converter.model = self.pt_model
        converter.eval_func = eval
        output_model = converter.fit()
        self.assertTrue(isinstance(output_model.model.fc, BF16ModuleWrapper))


if __name__ == "__main__":
    unittest.main()
