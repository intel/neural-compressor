import os
import unittest

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.tensorflow.utils import disable_random


def create_graph():
    input_node = node_def_pb2.NodeDef()
    input_node.name = "input"
    input_node.op = "Placeholder"
    input_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))

    const_node_1 = node_def_pb2.NodeDef()
    const_node_1.name = "const_1"
    const_node_1.op = "Const"
    const_value_1 = np.float32(np.random.randn(128))
    const_node_1.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    const_node_1.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(const_value_1, const_value_1.dtype.type, const_value_1.shape)
        )
    )

    const_node_2 = node_def_pb2.NodeDef()
    const_node_2.name = "const_2"
    const_node_2.op = "Const"
    const_value_2 = np.float32(np.random.randn(128))
    const_node_2.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    const_node_2.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(const_value_2, const_value_2.dtype.type, const_value_2.shape)
        )
    )

    const_node_3 = node_def_pb2.NodeDef()
    const_node_3.name = "const_3"
    const_node_3.op = "Const"
    const_value_3 = np.float32(np.random.randn(128))
    const_node_3.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    const_node_3.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(const_value_3, const_value_3.dtype.type, const_value_3.shape)
        )
    )

    const_node_4 = node_def_pb2.NodeDef()
    const_node_4.name = "const_4"
    const_node_4.op = "Const"
    const_value_4 = np.float32(np.random.randn(128))
    const_node_4.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    const_node_4.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(const_value_4, const_value_4.dtype.type, const_value_4.shape)
        )
    )

    pack_node = node_def_pb2.NodeDef()
    pack_node.name = "pack"
    pack_node.op = "Pack"
    pack_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    pack_node.attr["axis"].CopyFrom(attr_value_pb2.AttrValue(i=1))
    pack_node.attr["N"].CopyFrom(attr_value_pb2.AttrValue(i=4))
    pack_node.input.extend([const_node_1.name, const_node_2.name, const_node_3.name, const_node_4.name])

    shape_node = node_def_pb2.NodeDef()
    shape_node.name = "const_5"
    shape_node.op = "Const"
    value_4 = np.int32([1, 1, 128, 4])
    shape_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.qint32.as_datatype_enum))
    shape_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(value_4, value_4.dtype.type, value_4.shape))
    )

    reshape_node = node_def_pb2.NodeDef()
    reshape_node.name = "reshape"
    reshape_node.op = "Reshape"
    reshape_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    reshape_node.attr["Tshape"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum))
    reshape_node.input.extend([pack_node.name, shape_node.name])

    conv2_node = node_def_pb2.NodeDef()
    conv2_node.name = "conv"
    conv2_node.op = "Conv2D"
    conv2_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv2_node.input.extend([input_node.name, reshape_node.name])
    conv2_node.attr["strides"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv2_node.attr["dilations"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv2_node.attr["padding"].CopyFrom(attr_value_pb2.AttrValue(s=b"SAME"))
    conv2_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))
    test_graph = graph_pb2.GraphDef()

    test_graph.node.extend(
        [
            input_node,
            const_node_1,
            const_node_2,
            const_node_3,
            const_node_4,
            pack_node,
            shape_node,
            reshape_node,
            conv2_node,
        ]
    )
    return test_graph


class TestFetchWeightFromReshapeOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_graph = create_graph()

    @disable_random()
    def test_FetchWeightFromReshape_Optimizer(self):
        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.fetch_weight_from_reshape import (
            FetchWeightFromReshapeOptimizer,
        )

        convert_graph = FetchWeightFromReshapeOptimizer(self.test_graph).do_transformation()

        handled = False
        for node in convert_graph.node:
            if node.op == "Conv2D" and node.input[1] == "reshape/weight_0":
                handled = True
                break

        self.assertEqual(handled, True)


if __name__ == "__main__":
    unittest.main()
