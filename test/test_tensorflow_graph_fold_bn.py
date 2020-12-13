#
#  -*- coding: utf-8 -*-
#
import unittest
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes

from lpot.adaptor.tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper

from lpot.adaptor.tf_utils.graph_rewriter.generic.fold_batch_norm import \
    FoldBatchNormNodesOptimizer


class TestGraphFoldBNWithInvalidParameter(unittest.TestCase):
    def test_graph_cse(self):
        tf.compat.v1.disable_eager_execution()

        input_constant_name = "input_constant"
        relu_name = "relu"
        float_graph_def = graph_pb2.GraphDef()
        input_constant = QuantizeGraphHelper.create_constant_node(
            input_constant_name,
            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dtype=dtypes.float32,
            shape=[1, 2, 6, 1])
        float_graph_def.node.extend([input_constant])
        relu_node = QuantizeGraphHelper.create_node("Relu", relu_name,
                                                    [input_constant_name])
        QuantizeGraphHelper.set_attr_dtype(relu_node, "T", dtypes.float32)
        float_graph_def.node.extend([relu_node])

        b_constant_name = "b_constant"
        conv2d_name = "conv2d_1"
        b_constant = QuantizeGraphHelper.create_constant_node(
            b_constant_name,
            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dtype=dtypes.float32,
            shape=[1, 2, 3, 4])
        float_graph_def.node.extend([b_constant])

        conv2d_node = QuantizeGraphHelper.create_node(
            "Conv2D", conv2d_name, [relu_name, b_constant_name])
        QuantizeGraphHelper.set_attr_dtype(conv2d_node, "T", dtypes.float32)

        float_graph_def.node.extend([conv2d_node])

        bias_add_name = "bias_add"
        offset_constant_name = "offset_constant"

        offset_constant = QuantizeGraphHelper.create_constant_node(
            offset_constant_name,
            value=[1, 2, 3, 4, 5, 6],
            dtype=dtypes.float32,
            shape=[6])
        float_graph_def.node.extend([offset_constant])

        bias_add_node = QuantizeGraphHelper.create_node(
            "BiasAdd", bias_add_name, [conv2d_name, offset_constant_name])
        QuantizeGraphHelper.set_attr_dtype(bias_add_node, "T", dtypes.float32)
        float_graph_def.node.extend([bias_add_node])

        bn_scale_name = 'bn_scale'
        bn_scale_node = QuantizeGraphHelper.create_constant_node(
            bn_scale_name,
            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dtype=dtypes.float32,
            shape=[12, 1])
        bn_offset_name = 'bn_offset'
        bn_offset_node = QuantizeGraphHelper.create_constant_node(
            bn_offset_name,
            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dtype=dtypes.float32,
            shape=[12, 1])
        bn_mean_name = 'bn_mean'
        bn_mean_node = QuantizeGraphHelper.create_constant_node(
            bn_mean_name, value=[
                1,
                2,
            ], dtype=dtypes.float32, shape=[
                2,
            ])
        bn_var_name = 'bn_var'
        bn_var_node = QuantizeGraphHelper.create_constant_node(
            bn_var_name, value=[], dtype=dtypes.float32, shape=[0])
        fused_bn_node_name = 'bn'
        fused_bn_node = QuantizeGraphHelper.create_node(
            "FusedBatchNormV3", fused_bn_node_name, [
                bias_add_name, bn_scale_name, bn_offset_name, bn_mean_name,
                bn_var_name
            ])
        QuantizeGraphHelper.set_attr_dtype(fused_bn_node, "T", dtypes.float32)
        QuantizeGraphHelper.set_attr_dtype(fused_bn_node, "U", dtypes.float32)
        float_graph_def.node.extend([
            fused_bn_node, bn_scale_node, bn_offset_node, bn_mean_node,
            bn_var_node
        ])

        post_relu_name = "post_relu"
        post_relu_node = QuantizeGraphHelper.create_node(
            "Relu", post_relu_name, [fused_bn_node_name])
        float_graph_def.node.extend([post_relu_node])

        post_graph = FoldBatchNormNodesOptimizer(
            float_graph_def).do_transformation()

        bn_not_fused = False
        for i in post_graph.node:
            if i.op == 'FusedBatchNormV3':
                bn_not_fused = True
                break

        self.assertEqual(bn_not_fused, True)


if __name__ == '__main__':
    unittest.main()
