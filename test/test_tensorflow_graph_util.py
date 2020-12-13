import unittest
import copy
import tensorflow as tf
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from lpot.adaptor.tf_utils.graph_rewriter.graph_util import GraphAnalyzer, GraphRewriterHelper


class TestGraph_util(unittest.TestCase):
    x_node = node_def_pb2.NodeDef()
    x_node.name = "placeholder"
    x_node.op = "Placeholder"

    input0_node = node_def_pb2.NodeDef()
    input0_node.name = "input0"
    input0_node.op = "Const"
    input0_value = np.float32(np.abs(np.random.randn(4, 3, 2)))
    input0_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            input0_value, input0_value.dtype.type, input0_value.shape)))

    input1_node = node_def_pb2.NodeDef()
    input1_node.name = "input1"
    input1_node.op = "Const"
    input1_value = np.float32(np.abs(np.random.randn(4, 1, 1)))
    input1_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            input1_value, input1_value.dtype.type, input1_value.shape)))

    add_node = node_def_pb2.NodeDef()
    add_node.op = "Add"
    add_node.name = "add"
    add_node.input.extend([input0_node.name, input1_node.name])

    input2_node = node_def_pb2.NodeDef()
    input2_node.name = "input2"
    input2_node.op = "Const"
    input2_value = np.float32(np.abs(np.random.randn(1)))
    input2_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            input2_value, input2_value.dtype.type, input2_value.shape)))

    input3_node = node_def_pb2.NodeDef()
    input3_node.name = "input3"
    input3_node.op = "Const"
    input3_value = np.float32(np.abs(np.random.randn(1)))
    input3_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            input3_value, input3_value.dtype.type, input3_value.shape)))

    mul_node = node_def_pb2.NodeDef()
    mul_node.op = "Mul"
    mul_node.name = "mul"
    mul_node.input.extend([add_node.name, input3_node.name])

    sqrt_node = node_def_pb2.NodeDef()
    sqrt_node.name = "rsqrt"
    sqrt_node.op = "Rsqrt"
    sqrt_node.input.extend([mul_node.name])

    sqrt1_node = node_def_pb2.NodeDef()
    sqrt1_node.op = "Relu"
    sqrt1_node.name = "sqrt1"
    sqrt1_node.input.extend([sqrt_node.name])

    block_node = node_def_pb2.NodeDef()
    block_node.name = "block_output"
    block_node.op = "Add"
    block_node.input.extend([x_node.name, sqrt1_node.name])

    res_node = node_def_pb2.NodeDef()
    res_node.name = "res_add"
    res_node.op = "Add"
    res_node.input.extend([sqrt_node.name, input2_node.name])

    end_node = node_def_pb2.NodeDef()
    end_node.name = "end"
    end_node.op = "Add"
    end_node.input.extend([block_node.name, res_node.name])

    graph_def = graph_pb2.GraphDef()
    graph_def.node.extend([
        x_node, input0_node, input1_node, input2_node, input3_node, add_node, mul_node, sqrt_node,
        sqrt1_node, block_node, res_node, end_node
    ])

    def test_replace_constant_graph_with_constant_node(self):
        graph_analyzer = GraphAnalyzer()
        graph_analyzer.graph = copy.deepcopy(self.graph_def)

        graph_analyzer.parse_graph()

        new_constant_value = np.random.random([4, 1])
        new_constant_type = tf.as_dtype(np.float32(new_constant_value).dtype)
        new_constant_node = GraphRewriterHelper.create_constant_node(
            self.add_node.name + "_const", new_constant_value, new_constant_type)
        assert graph_analyzer.replace_constant_graph_with_constant_node(
            new_constant_node, self.add_node.name)
        result_graph = graph_analyzer.dump_graph()
        assert len(list(result_graph.node)) == 10

        new_constant_value = np.random.random([4, 1])
        new_constant_type = tf.as_dtype(np.float32(new_constant_value).dtype)
        new_constant_node = GraphRewriterHelper.create_constant_node(
            self.mul_node.name + "_const", new_constant_value, new_constant_type)
        assert graph_analyzer.replace_constant_graph_with_constant_node(
            new_constant_node, self.mul_node.name)
        result_graph = graph_analyzer.dump_graph()
        assert len(list(result_graph.node)) == 8

        new_constant_value = np.random.random([4, 1])
        new_constant_type = tf.as_dtype(np.float32(new_constant_value).dtype)
        new_constant_node = GraphRewriterHelper.create_constant_node(
            self.sqrt_node.name + "_const", new_constant_value, new_constant_type)
        assert graph_analyzer.replace_constant_graph_with_constant_node(
            new_constant_node, self.sqrt_node.name)
        result_graph = graph_analyzer.dump_graph()
        assert len(list(result_graph.node)) == 7

        new_constant_value = np.random.random([4, 1])
        new_constant_type = tf.as_dtype(np.float32(new_constant_value).dtype)
        new_constant_node = GraphRewriterHelper.create_constant_node(
            self.block_node.name + "_const", new_constant_value, new_constant_type)
        assert not graph_analyzer.replace_constant_graph_with_constant_node(
            new_constant_node, self.block_node.name)

    def test_replace_node(self):
        graph_analyzer = GraphAnalyzer()
        graph_analyzer.graph = copy.deepcopy(self.graph_def)

        graph_analyzer.parse_graph()

        new_add_node = node_def_pb2.NodeDef()
        new_add_node.op = "Add"
        new_add_node.name = "add1"
        new_add_node.input.extend([self.input0_node.name, self.input1_node.name])
        graph_analyzer.replace_node(new_add_node, self.add_node.name, [self.mul_node.name])
        result_graph = graph_analyzer.dump_graph()
        assert self.add_node not in list(result_graph.node)
        assert new_add_node in list(result_graph.node)


if __name__ == "__main__":
    unittest.main()
