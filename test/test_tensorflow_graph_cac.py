#
#  -*- coding: utf-8 -*-
#
import unittest
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from neural_compressor.adaptor.tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.scale_propagation import \
    ScaleProPagationTransformer


class TestGraphScaleProPagation(unittest.TestCase):
    def test_scale_propagation(self):
        """Test scale propagation for below pattern
        requantize + quantizedavgpool+ quantized conv2d + requantize.
        """
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
        requantize_min_name = "requantize_min_const"
        requantize_min = QuantizeGraphHelper.create_constant_node(
            requantize_min_name,
            value=1,
            dtype=dtypes.float32,
        )
        float_graph_def.node.extend([requantize_min])
        requantize_max_name = "requantize_max_const"
        requantize_max = QuantizeGraphHelper.create_constant_node(
            requantize_max_name,
            value=5,
            dtype=dtypes.float32,
        )
        float_graph_def.node.extend([requantize_max])
        relu_node = QuantizeGraphHelper.create_node("Requantize", relu_name, [
            input_constant_name, input_constant_name + ':1',
            input_constant_name + ':2', requantize_min_name,
            requantize_max_name
        ])
        QuantizeGraphHelper.set_attr_dtype(relu_node, "Tinput", dtypes.qint32)
        QuantizeGraphHelper.set_attr_dtype(relu_node, "out_type",
                                           dtypes.quint8)
        float_graph_def.node.extend([relu_node])

        b_constant_name = "b_constant"
        mat_mul_name = "mat_mul"
        b_constant = QuantizeGraphHelper.create_constant_node(
            b_constant_name, value=[0], dtype=dtypes.float32, shape=[
                1,
            ])
        float_graph_def.node.extend([b_constant])

        avgpool_max_constant_name = "avgpool_max_constant"
        mat_mul_name = "mat_mul"
        avgpool_max = QuantizeGraphHelper.create_constant_node(
            avgpool_max_constant_name,
            value=[10],
            dtype=dtypes.float32,
            shape=[
                1,
            ])
        float_graph_def.node.extend([avgpool_max])
        quantized_avgpool = QuantizeGraphHelper.create_node(
            "QuantizedAvgPool", mat_mul_name,
            [relu_name, b_constant_name, avgpool_max_constant_name])
        QuantizeGraphHelper.set_attr_dtype(quantized_avgpool, "T", dtypes.float32)

        float_graph_def.node.extend([quantized_avgpool])

        bias_add_name = "bias_add"
        offset_constant_name = "offset_constant"

        offset_constant = QuantizeGraphHelper.create_constant_node(
            offset_constant_name,
            value=[1, 2, 3, 4, 5, 6],
            dtype=dtypes.float32,
            shape=[6])
        float_graph_def.node.extend([offset_constant])
        bias_add_node = QuantizeGraphHelper.create_node(
            "QuantizedConv2DWithBiasAndRelu", bias_add_name,
            [mat_mul_name, offset_constant_name])
        QuantizeGraphHelper.set_attr_dtype(bias_add_node, "T", dtypes.float32)
        float_graph_def.node.extend([bias_add_node])
        post_min_value = -1
        post_max_value = 7
        post_requantize_min_name = "post_requantize_min_const"
        post_requantize_min = QuantizeGraphHelper.create_constant_node(
            post_requantize_min_name,
            value=post_min_value,
            dtype=dtypes.float32,
        )
        float_graph_def.node.extend([post_requantize_min])
        post_requantize_max_name = "post_requantize_max_const"
        post_requantize_max = QuantizeGraphHelper.create_constant_node(
            post_requantize_max_name,
            value=post_max_value,
            dtype=dtypes.float32,
        )
        float_graph_def.node.extend([post_requantize_max])

        post_requantize_name = "post_requantize"
        post_requantize_node = QuantizeGraphHelper.create_node(
            "Requantize", post_requantize_name, [
                bias_add_name, bias_add_name + ':1', bias_add_name + ':2',
                post_requantize_min_name, post_requantize_max_name
            ])
        float_graph_def.node.extend([post_requantize_node])

        optimized_graph = ScaleProPagationTransformer(
            float_graph_def).do_transformation()
        update_min_value = None
        update_max_value = None
        for node in optimized_graph.node:
            if node.name == 'relu_cac_requantize_min_value':
                update_min_value = node.attr['value'].tensor.float_val[0]

            if node.name == 'relu_cac_requantize_max_value':
                update_max_value = node.attr['value'].tensor.float_val[0]

        self.assertEqual(update_min_value, post_min_value)
        self.assertEqual(update_max_value, post_max_value)


if __name__ == '__main__':
    unittest.main()
