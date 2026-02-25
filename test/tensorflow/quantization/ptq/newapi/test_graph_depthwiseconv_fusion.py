#
#  -*- coding: utf-8 -*-
#
import os
import unittest

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


def build_Conv2dBiasAddAddRelu6MulMul():
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
    bias_add_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))
    bias_add_node.input.extend([conv1_node.name, bias_node.name])

    offset_node = node_def_pb2.NodeDef()
    offset_node.name = "offset"
    offset_node.op = "Const"
    offset_value = np.float32(np.abs(np.random.randn(1)))
    offset_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    offset_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(offset_value, offset_value.dtype.type, offset_value.shape)
        )
    )

    add_node = node_def_pb2.NodeDef()
    add_node.op = "Add"
    add_node.name = "add/hard_swish"
    add_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    add_node.input.extend([bias_add_node.name, offset_node.name])

    relu_node = node_def_pb2.NodeDef()
    relu_node.op = "Relu6"
    relu_node.name = "relu6/hard_swish"
    relu_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    relu_node.input.extend([add_node.name])

    mul_node = node_def_pb2.NodeDef()
    mul_node.op = "Mul"
    mul_node.name = "mul/hard_swish"
    mul_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    mul_node.input.extend([bias_add_node.name, relu_node.name])

    offset1_node = node_def_pb2.NodeDef()
    offset1_node.name = "mul1_offset"
    offset1_node.op = "Const"
    offset1_value = np.float32(np.abs(np.random.randn(1)))
    offset1_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    offset1_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(offset1_value, offset1_value.dtype.type, offset1_value.shape)
        )
    )

    mul1_node = node_def_pb2.NodeDef()
    mul1_node.op = "Mul"
    mul1_node.name = "mul1/hard_swish"
    mul1_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    mul1_node.input.extend([mul_node.name, offset1_node.name])

    test_graph = graph_pb2.GraphDef()

    test_graph.node.extend(
        [
            input_node,
            conv1_weight_node,
            conv1_node,
            bias_node,
            bias_add_node,
            add_node,
            relu_node,
            offset_node,
            offset1_node,
            mul_node,
            mul1_node,
        ]
    )
    return test_graph


class TestConvBiasAddAddReluFusion(unittest.TestCase):
    @disable_random()
    def test_depthwiseconv_biasadd_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.depthwise_conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="VALID")

        normed = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv, name="op_to_store")
        )
        out_name = normed.name.split(":")[0]

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 56, 56, 16), label=True)
            calib_dataloader = BaseDataLoader(dataset)
            quant_config = {
                "static_quant": {
                    "global": {
                        "weight_dtype": "int8",
                        "weight_sym": True,
                        "weight_granularity": "per_tensor",
                        "weight_algorithm": "minmax",
                    },
                }
            }
            qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedDepthwiseConv2D":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_depthwiseConv2dNative_BiasAddAddRelu6MulMul_fusion(self):
        fp32_graph_def = build_Conv2dBiasAddAddRelu6MulMul()

        from neural_compressor.tensorflow import quantize_model
        from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

        dataset = DummyDataset(shape=(100, 224, 224, 3), label=True)
        calib_dataloader = BaseDataLoader(dataset)
        quant_config = {
            "static_quant": {
                "global": {
                    "weight_dtype": "int8",
                    "weight_sym": True,
                    "weight_granularity": "per_tensor",
                    "weight_algorithm": "minmax",
                },
            }
        }
        qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

        found_conv_fusion = False
        for i in qmodel.graph_def.node:
            if i.op == "_FusedQuantizedConv2D":
                found_conv_fusion = True
                break

        self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_depthwiseconv_biasadd_leakyrelu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.depthwise_conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="VALID")

        normed = (
            tf.keras.layers.BatchNormalization(name="op_to_store")(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv, name="op_to_store")
        )

        leakyrelu = tf.nn.leaky_relu(normed)
        out_name = leakyrelu.name.split(":")[0]

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 56, 56, 16), label=True)
            calib_dataloader = BaseDataLoader(dataset)
            quant_config = {
                "static_quant": {
                    "global": {
                        "weight_dtype": "int8",
                        "weight_sym": True,
                        "weight_granularity": "per_tensor",
                        "weight_algorithm": "minmax",
                    },
                }
            }
            qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

            found_conv_fusion = False
            for i in qmodel.graph_def.node:
                if i.op == "_FusedQuantizedDepthwiseConv2D":
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)


if __name__ == "__main__":
    unittest.main()
