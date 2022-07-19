import unittest
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fold_constant import GraphFoldConstantOptimizer


class TestFoldConstant(unittest.TestCase):
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

    switch_node = node_def_pb2.NodeDef()
    switch_node.name = "switch"
    switch_node.op = "Switch"

    input4_node = node_def_pb2.NodeDef()
    input4_node.name = "input4"
    input4_node.op = "Const"
    input4_value = np.float32(np.abs(np.random.randn(1)))
    input4_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            input4_value, input4_value.dtype.type, input4_value.shape)))
    input4_node.input.extend([switch_node.name])

    input5_node = node_def_pb2.NodeDef()
    input5_node.name = "input5"
    input5_node.op = "Const"
    input5_value = np.float32(np.abs(np.random.randn(1)))
    input5_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            input5_value, input5_value.dtype.type, input5_value.shape)))
    input5_node.input.extend([switch_node.name])

    cond_end = node_def_pb2.NodeDef()
    cond_end.name = "cond"
    cond_end.op = "Add"
    cond_end.input.extend([input4_node.name, input5_node.name])

    mul_node = node_def_pb2.NodeDef()
    mul_node.op = "Mul"
    mul_node.name = "mul"
    mul_node.input.extend([add_node.name, input3_node.name])

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
    res_node.input.extend([sqrt_node.name, input2_node.name])

    end_node = node_def_pb2.NodeDef()
    end_node.name = "end"
    end_node.op = "Add"
    end_node.input.extend([block_node.name, res_node.name])

    graph_def = graph_pb2.GraphDef()
    graph_def.node.extend([
        x_node, input0_node, input1_node, input2_node, input3_node, add_node, mul_node, sqrt_node,
        relu_node, block_node, res_node, end_node
    ])

    def test_fold_constant(self):

        graph = self.graph_def
        rewriter = GraphFoldConstantOptimizer(graph)
        new_graph = rewriter.do_transformation()

        for node in new_graph.node:
            assert node.name in [
                "placeholder", "block_output", "rsqrt_const", "relu", "res_add_const", "end"
            ]

    def test_condition_fold_constant(self):
        graph_def = graph_pb2.GraphDef()
        graph_def.node.extend([self.cond_end, self.input4_node,
                               self.input5_node, self.switch_node])
        rewriter = GraphFoldConstantOptimizer(graph_def)
        new_graph = rewriter.do_transformation()
        for node in new_graph.node:
            assert node.name in ["switch", "cond", "input4", "input5"]

    def test_slice_int_input(self):
        graph_def = graph_pb2.GraphDef()
        index0_node = node_def_pb2.NodeDef()
        index0_node.name = "index0"
        index0_node.op = "Const"
        index0_value = np.array(3).astype(np.int32).reshape(())
        index0_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            index0_value, index0_value.dtype.type, index0_value.shape)))

        index1_node = node_def_pb2.NodeDef()
        index1_node.name = "index1"
        index1_node.op = "Const"
        index1_value = np.array(1).astype(np.int32).reshape(())
        index1_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            index1_value, index1_value.dtype.type, index1_value.shape)))

        minus_node = node_def_pb2.NodeDef()
        minus_node.name = "sub"
        minus_node.op = "Sub"
        minus_node.input.extend([index0_node.name, index1_node.name])

        graph_def.node.extend([index0_node, index1_node, minus_node])
        rewriter = GraphFoldConstantOptimizer(graph_def)
        new_graph = rewriter.do_transformation()
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.import_graph_def(new_graph)

if __name__ == "__main__":
    unittest.main()
