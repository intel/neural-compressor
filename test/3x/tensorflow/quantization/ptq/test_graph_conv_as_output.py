import os
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util


def create_test_graph():
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

    log_node = node_def_pb2.NodeDef()
    log_node.name = "log1"
    log_node.op = "Log"
    log_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    log_node.input.extend([relu_node2.name])

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
    conv3_node.input.extend([log_node.name, conv3_weight_node.name])
    conv3_node.attr["strides"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv3_node.attr["dilations"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv3_node.attr["padding"].CopyFrom(attr_value_pb2.AttrValue(s=b"SAME"))
    conv3_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    test_graph = graph_pb2.GraphDef()

    test_graph.node.extend(
        [
            input_node,
            conv1_weight_node,
            conv1_node,
            bias_node,
            bias_add_node,
            relu_node,
            conv2_weight_node,
            conv2_node,
            bias_node2,
            bias_add_node2,
            log_node,
            relu_node2,
            conv3_weight_node,
            conv3_node,
        ]
    )
    return test_graph


@unittest.skipIf(tf.__version__ < "2.8.0", "only support spr-base TF")
class TestConvAsOutput(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_graph = create_test_graph()

    def test_do_transform(self):
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
        from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

        fp32_graph_def = create_test_graph()
        dataset = DummyDataset(shape=(1, 224, 224, 3), label=True)
        calib_dataloader = BaseDataLoader(dataset)
        quant_config = StaticQuantConfig()
        qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

        for node in qmodel.graph_def.node:
            if node.name == "conv3_eightbit_requantize":
                self.assertTrue("Quantized" in node.op)


if __name__ == "__main__":
    unittest.main()
