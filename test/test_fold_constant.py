import unittest
import tensorflow as tf
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from ilit.adaptor.tf_utils.transform_graph.fold_constant import FoldConstant


class TestFoldConst(unittest.TestCase):
    x_node = node_def_pb2.NodeDef()
    x_node.name = "placeholder"
    x_node.op = "Placeholder"

    input0_node = node_def_pb2.NodeDef()
    input0_node.name = "input0"
    input0_node.op = "Const"
    input0_value = np.float32(np.abs(np.random.randn(4, 3, 2)))
    input0_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
        input0_value, input0_value.dtype.type, input0_value.shape)))

    input0_read_node = node_def_pb2.NodeDef()
    input0_read_node.name = "read0"
    input0_read_node.op = "Identity"
    input0_read_node.input.extend([input0_node.name])

    input1_node = node_def_pb2.NodeDef()
    input1_node.name = "input1"
    input1_node.op = "Const"
    input1_value = np.float32(np.abs(np.random.randn(4, 1, 1)))
    input1_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
        input1_value, input1_value.dtype.type, input1_value.shape)))

    input1_read_node = node_def_pb2.NodeDef()
    input1_read_node.name = "read1"
    input1_read_node.op = "Identity"
    input1_read_node.input.extend([input1_node.name])

    add_node = node_def_pb2.NodeDef()
    add_node.op = "Add"
    add_node.name = "add"
    add_node.input.extend([input0_read_node.name, input1_read_node.name])

    input2_node = node_def_pb2.NodeDef()
    input2_node.name = "input2"
    input2_node.op = "Const"
    input2_value = np.float32(np.abs(np.random.randn(1)))
    input2_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
        input2_value, input2_value.dtype.type, input2_value.shape)))

    input2_read_node = node_def_pb2.NodeDef()
    input2_read_node.name = "read2"
    input2_read_node.op = "Identity"
    input2_read_node.input.extend([input2_node.name])

    mul_node = node_def_pb2.NodeDef()
    mul_node.op = "Mul"
    mul_node.name = "mul"
    mul_node.input.extend([add_node.name, input2_read_node.name])

    sqrt_node = node_def_pb2.NodeDef()
    sqrt_node.name = "rsqrt"
    sqrt_node.op = "Rsqrt"
    sqrt_node.input.extend([mul_node.name])

    relu_node = node_def_pb2.NodeDef()
    relu_node.op = "Relu"
    relu_node.name = "relu"
    relu_node.input.extend([sqrt_node.name])

    block_node = node_def_pb2.NodeDef()
    block_node.name = "block_output"
    block_node.op = "Add"
    block_node.input.extend([x_node.name, relu_node.name])

    res_node = node_def_pb2.NodeDef()
    res_node.name = "res_add"
    res_node.op = "Add"
    res_node.input.extend([sqrt_node.name, input2_read_node.name])

    end_node = node_def_pb2.NodeDef()
    end_node.name = "end"
    end_node.op = "Add"
    end_node.input.extend([block_node.name, res_node.name])

    graph_def = graph_pb2.GraphDef()
    graph_def.node.extend([x_node,
                           input0_node,
                           input1_node,
                           input2_node,
                           input0_read_node,
                           input1_read_node,
                           input2_read_node,
                           add_node,
                           mul_node,
                           sqrt_node,
                           relu_node,
                           block_node,
                           res_node,
                           end_node])

    def test_fold_values(self):
        graph = self.graph_def
        transformer = FoldConstant(graph)

        end_node_name = self.sqrt_node.name
        fold_value = transformer._fold_value(end_node_name)
        target_value = 1 / np.sqrt((self.input0_value + self.input1_value) * self.input2_value)
        assert np.allclose(fold_value, target_value)

        end_node_name = self.res_node.name
        fold_value = transformer._fold_value(end_node_name)
        target_value = 1 / np.sqrt((self.input0_value + self.input1_value)
                                   * self.input2_value) + self.input2_value
        assert np.allclose(fold_value, target_value)

    def test_do_transform(self):
        graph = self.graph_def
        transformer = FoldConstant(graph)
        new_graph = transformer.do_transformation(["placeholder"], ["end"])

        assert len(transformer.end_nodes) == 2
        assert self.sqrt_node.name in transformer.end_nodes
        assert self.res_node.name in transformer.end_nodes

        for node in new_graph.node:
            assert node.name in [
                "placeholder",
                "block_output",
                "rsqrt",
                "relu",
                "res_add",
                "end"]


if __name__ == "__main__":
    unittest.main()
