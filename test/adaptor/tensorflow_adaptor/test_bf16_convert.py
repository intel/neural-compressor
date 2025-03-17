import os
import platform
import shutil
import unittest
from unittest import result

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.adaptor.tf_utils.graph_rewriter.bf16.bf16_convert import BF16Convert


def create_test_graph(bf16_graph=True):
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

    if bf16_graph:
        cast_node = node_def_pb2.NodeDef()
        cast_node.op = "Cast"
        cast_node.name = "cast"
        cast_node.attr["SrcT"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
        cast_node.attr["DstT"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.bfloat16.as_datatype_enum))
        cast_node.input.extend([bias_add_node.name])

    relu_node = node_def_pb2.NodeDef()
    relu_node.op = "Relu"
    relu_node.name = "relu"
    relu_node.attr["T"].CopyFrom(
        attr_value_pb2.AttrValue(
            type=dtypes.bfloat16.as_datatype_enum if bf16_graph else dtypes.float32.as_datatype_enum
        )
    )
    relu_node.input.extend([cast_node.name if bf16_graph else bias_add_node.name])

    if bf16_graph:
        cast2_node = node_def_pb2.NodeDef()
        cast2_node.op = "Cast"
        cast2_node.name = "cast2"
        cast2_node.attr["SrcT"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.bfloat16.as_datatype_enum))
        cast2_node.attr["DstT"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
        cast2_node.input.extend([relu_node.name])

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
    conv2_node.input.extend([cast2_node.name if bf16_graph else relu_node.name, conv2_weight_node.name])
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

    identity_node = node_def_pb2.NodeDef()
    identity_node.name = "final"
    identity_node.op = "Identity"
    identity_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    identity_node.input.extend([conv3_node.name])

    test_graph = graph_pb2.GraphDef()

    if bf16_graph:
        test_graph.node.extend(
            [
                input_node,
                conv1_weight_node,
                conv1_node,
                bias_node,
                bias_add_node,
                cast_node,
                relu_node,
                cast2_node,
                conv2_weight_node,
                conv2_node,
                bias_node2,
                bias_add_node2,
                log_node,
                relu_node2,
                conv3_weight_node,
                conv3_node,
                identity_node,
            ]
        )
    else:
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
                identity_node,
            ]
        )
    return test_graph


class TestBF16Convert(unittest.TestCase):
    rn50_fp32_pb_url = (
        "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb"
    )
    pb_path = "/tmp/.neural_compressor/resnet50_fp32_pretrained_model.pb"
    platform = platform.system().lower()
    if platform == "windows":
        pb_path = "C:\\tmp\.neural_compressor\\resnet50_fp32_pretrained_model.pb"

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path):
            if self.platform == "linux":
                os.system(
                    "mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(self.rn50_fp32_pb_url, self.pb_path)
                )
            elif self.platform == "windows":
                os.system("md C:\\tmp\.neural_compressor && cd C:\\tmp\.neural_compressor")
                from urllib import request

                request.urlretrieve(self.rn50_fp32_pb_url)

        self.input_graph = tf.compat.v1.GraphDef()
        with open(self.pb_path, "rb") as f:
            self.input_graph.ParseFromString(f.read())
        self.test_graph = create_test_graph()
        self.test_fp32_graph = create_test_graph(False)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("saved", ignore_errors=True)

    @unittest.skipIf(tf.__version__ < "2.0", "currently bf16 convert does not support 1.15up3")
    def test_rn50_convert(self):
        bf16_nodes = [node.name for node in self.input_graph.node if node.op in ["Conv2D", "AvgPool", "MatMul"]]
        bf16_nodes.remove("v0/resnet_v13/conv14/conv2d/Conv2D")
        rn50_bf16_converter = BF16Convert(self.input_graph, ["v0/resnet_v13/conv14/conv2d/Conv2D"], bf16_nodes)
        rn50_bf16_converter.do_transformation()
        new_conv11 = rn50_bf16_converter.cur_graph.node_name_details["v0/resnet_v13/conv11/conv2d/Conv2D"].node
        new_conv14 = rn50_bf16_converter.cur_graph.node_name_details["v0/resnet_v13/conv14/conv2d/Conv2D"].node
        new_conv52 = rn50_bf16_converter.cur_graph.node_name_details["v0/resnet_v115/conv52/conv2d/Conv2D"].node
        self.assertEqual(new_conv11.attr["T"].type, new_conv52.attr["T"].type)
        self.assertNotEqual(new_conv11.attr["T"].type, new_conv14.attr["T"].type)

    @unittest.skipIf(tf.__version__ < "2.0", "currently bf16 convert does not support 1.15up3")
    def test_do_transform(self):
        bf16_converter = BF16Convert(self.test_graph, ["conv3"], ["conv2", "relu2"])
        new_graph = bf16_converter.do_transformation()
        new_conv2 = bf16_converter.cur_graph.node_name_details["conv2"].node
        new_conv3 = bf16_converter.cur_graph.node_name_details["conv3"].node
        new_relu2 = bf16_converter.cur_graph.node_name_details["relu2"].node
        self.assertEqual(new_conv2.attr["T"].type, dtypes.bfloat16)
        self.assertEqual(new_relu2.attr["T"].type, dtypes.bfloat16)
        self.assertEqual(new_conv3.attr["T"].type, dtypes.float32)


if __name__ == "__main__":
    unittest.main()
