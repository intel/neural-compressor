import os
import unittest

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.python.framework import dtypes

from neural_compressor.tensorflow.quantization.utils.graph_rewriter.bf16.dequantize_cast_optimizer import (
    DequantizeCastOptimizer,
)
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.tensorflow.utils import disable_random


def build_fake_graphdef(set_min_first=False, dq_multi_outputs=False):
    tf.compat.v1.disable_eager_execution()

    input = tf.compat.v1.placeholder(tf.float32, shape=(32, 224, 224, 3), name="input")
    graph_def = tf.compat.v1.get_default_graph().as_graph_def(add_shapes=True)

    min_input = Helper.create_constant_node("test_min", value=0.0, dtype=dtypes.float32)

    max_input = Helper.create_constant_node("test_max", value=[1], dtype=dtypes.float32)

    quant_v2_node = Helper.create_node("QuantizeV2", "test_quantize", [input.name, min_input.name, max_input.name])

    dequantize_node = Helper.create_node(
        "Dequantize", "test_dequantize", [quant_v2_node.name, quant_v2_node.name + ":1", quant_v2_node.name + ":2"]
    )
    if set_min_first:
        Helper.set_attr_string(dequantize_node, "mode", b"MIN_FIRST")

    cast_node = Helper.create_node("Cast", "test_cast", [dequantize_node.name])
    Helper.set_attr_dtype(cast_node, "DstT", dtypes.bfloat16)
    Helper.set_attr_dtype(cast_node, "SrcT", dtypes.float32)
    Helper.set_attr_bool(cast_node, "Truncate", False)

    dentity_node = Helper.create_node("Identity", "output", [cast_node.name])
    Helper.set_attr_dtype(dentity_node, "T", dtypes.bfloat16)

    graph_def.node.extend(
        [
            min_input,
            max_input,
            quant_v2_node,
            dequantize_node,
            cast_node,
            dentity_node,
        ]
    )

    if dq_multi_outputs:
        dentity_node_2 = Helper.create_node("Identity", "id_1", [dequantize_node.name])
        Helper.set_attr_dtype(dentity_node_2, "T", dtypes.float32)
        graph_def.node.extend([dentity_node_2])

    return graph_def


class TestDequantizeCastOptimizer(unittest.TestCase):
    @disable_random()
    def test_dequantize_cast_normal(self):
        graph_def = build_fake_graphdef()
        converted_graph_def = DequantizeCastOptimizer(graph_def).do_transformation()
        for i in converted_graph_def.node:
            if i.op == "Cast":
                hasCast = True
                break

        self.assertEqual(hasCast, True)

    @disable_random()
    def test_dequantize_cast_min_first(self):
        graph_def = build_fake_graphdef(set_min_first=True)
        converted_graph_def = DequantizeCastOptimizer(graph_def).do_transformation()
        hasCast = False
        for i in converted_graph_def.node:
            if i.op == "Cast":
                hasCast = True
                break

        self.assertEqual(hasCast, True)

    @disable_random()
    def test_dequantize_cast_multiple_outputs(self):
        graph_def = build_fake_graphdef(dq_multi_outputs=True)
        converted_graph_def = DequantizeCastOptimizer(graph_def).do_transformation()
        hasCast = False
        for i in converted_graph_def.node:
            if i.op == "Cast":
                hasCast = True
                break

        self.assertEqual(hasCast, True)


if __name__ == "__main__":
    unittest.main()
