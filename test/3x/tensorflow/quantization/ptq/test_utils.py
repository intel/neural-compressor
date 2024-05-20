import os
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.tensorflow import Model
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.tensorflow.quantization.utils.utility import (
    collate_tf_preds,
    fix_ref_type_of_graph_def,
    generate_feed_dict,
    get_graph_def,
    get_model_input_shape,
    get_tensor_by_name,
    is_ckpt_format,
)
from neural_compressor.tensorflow.utils import disable_random


def build_fake_graphdef():
    graph_def = graph_pb2.GraphDef()
    constant_1_name = "moving_1/switch_input_const"
    constant_1 = Helper.create_constant_node(constant_1_name, value=0.0, dtype=dtypes.float32)

    constant_3_name = "moving_1/switch_input_const/read"
    constant_3 = Helper.create_constant_node(constant_3_name, value=[1], dtype=dtypes.float32)

    constant_2_name = "switch_input_const2"
    constant_2 = Helper.create_constant_node(constant_2_name, value=2.0, dtype=dtypes.float32)
    equal_name = "equal"
    equal = Helper.create_node("Equal", equal_name, [constant_1_name, constant_2_name])
    Helper.set_attr_dtype(equal, "T", dtypes.float32)

    refswitch_name = "refswitch"
    refswitch_node = Helper.create_node("RefSwitch", refswitch_name, [constant_1_name, equal_name])
    Helper.set_attr_dtype(refswitch_node, "T", dtypes.float32)

    variable_name = "variable"
    variable_node = Helper.create_node("VariableV2", variable_name, [])
    Helper.set_attr_dtype(variable_node, "T", dtypes.float32)

    assign_name = "assign"
    assign_node = Helper.create_node("Assign", assign_name, [variable_name, refswitch_name])
    Helper.set_attr_bool(assign_node, "use_locking", True)
    Helper.set_attr_bool(assign_node, "validate_shape", True)
    Helper.set_attr_dtype(assign_node, "T", dtypes.float32)

    assignsub_name = "assignsub"
    assignsub_node = Helper.create_node("AssignSub", assignsub_name, [assign_name, constant_1_name])
    Helper.set_attr_bool(assignsub_node, "use_locking", True)
    Helper.set_attr_dtype(assignsub_node, "T", dtypes.float32)

    assignadd_name = "assignadd"
    assignadd_node = Helper.create_node("AssignAdd", assignadd_name, [assignsub_name, constant_2_name])
    Helper.set_attr_bool(assignadd_node, "use_locking", True)
    Helper.set_attr_dtype(assignadd_node, "T", dtypes.float32)

    graph_def.node.extend(
        [
            constant_1,
            constant_2,
            constant_3,
            equal,
            refswitch_node,
            variable_node,
            assign_node,
            assignsub_node,
            assignadd_node,
        ]
    )
    return graph_def


def build_fake_graphdef2():
    input_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(32, 224, 224, 3), name="input_placeholder")

    conv_filter = tf.Variable(tf.random.normal([3, 3, 3, 32], stddev=0.1), name="conv_filter")
    conv_bias = tf.Variable(tf.zeros([32]), name="conv_bias")
    conv_output = tf.nn.conv2d(input_placeholder, conv_filter, strides=[1, 1, 1, 1], padding="SAME")
    conv_output = tf.nn.bias_add(conv_output, conv_bias)
    conv_output = tf.nn.relu(conv_output, name="conv_output")

    pool_output = tf.nn.max_pool2d(
        conv_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool_output"
    )

    fc_weights = tf.Variable(
        tf.random.normal(
            [int(pool_output.shape[1]) * int(pool_output.shape[2]) * int(pool_output.shape[3]), 10], stddev=0.1
        ),
        name="fc_weights",
    )
    fc_bias = tf.Variable(tf.zeros([10]), name="fc_bias")
    fc_input = tf.reshape(pool_output, [-1, int(fc_weights.shape[0])])
    fc_output = tf.matmul(fc_input, fc_weights) + fc_bias

    output = tf.reduce_sum(tf.nn.softmax(fc_output, name="output"), axis=-1)
    graph_def = tf.compat.v1.get_default_graph().as_graph_def()

    return graph_def


class TestTFutil(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        os.remove("test.pb")
        os.removedirs("fake_ckpt")

    @disable_random()
    def test_fix_ref_type(self):
        graph_def = build_fake_graphdef()
        new_graph_def = fix_ref_type_of_graph_def(graph_def)
        f = tf.io.gfile.GFile("./test.pb", "wb")
        f.write(new_graph_def.SerializeToString())
        find_Assign_prefix = False
        for node in new_graph_def.node:
            if "Assign" in node.op:
                find_Assign_prefix = True

        self.assertFalse(find_Assign_prefix, False)

    @disable_random()
    def test_collate_tf_preds(self):
        results = [[1], [np.array([2])]]
        data = collate_tf_preds(results)

        self.assertEqual(data, [1, np.array([2])])

        results = [[np.array([2])], [[1]]]
        data = collate_tf_preds(results)

        self.assertEqual(data[0].all(), np.array([2, 1]).all())

    @disable_random()
    def test_get_graph_def(self):
        graph_def = get_graph_def("./test.pb", outputs="assignadd")

        self.assertIsInstance(graph_def, tf.compat.v1.GraphDef)

    @disable_random()
    def test_judge_ckpt_format(self):
        os.mkdir("fake_ckpt")
        ckpt_format = is_ckpt_format("fake_ckpt")

        self.assertEqual(ckpt_format, False)

    @disable_random()
    def test_get_model_input_shape(self):
        graph_def = build_fake_graphdef2()
        try:
            tensor = get_tensor_by_name(graph_def, "fake:0")
        except:
            print("This code is for UT coverage of the exception handling")
        model = Model(graph_def)
        input_shape = get_model_input_shape(model)

        self.assertEqual(input_shape, 32)

    @disable_random()
    def test_generate_feed_dict(self):
        input_0 = [[1.0, 3.0], [3.0, 7.0]]
        input_tensor_0 = tf.convert_to_tensor(input_0)
        input_1 = [[1.0, 3.0]]
        input_tensor_1 = tf.convert_to_tensor(input_1)

        feed_dict = generate_feed_dict([input_tensor_0], input_0)
        self.assertEqual(feed_dict, {input_tensor_0: input_0})

        feed_dict = generate_feed_dict([input_tensor_0], {"Const": input_0})
        self.assertEqual(feed_dict, {input_tensor_0: input_0})

        feed_dict = generate_feed_dict([input_tensor_0, input_tensor_1], [input_0, input_1])
        self.assertEqual(feed_dict[input_tensor_0], input_0)
        self.assertEqual(feed_dict[input_tensor_1], input_1)

        feed_dict = generate_feed_dict([input_tensor_0, input_tensor_1], {"Const": input_0, "Const_1": input_1})
        self.assertEqual(feed_dict[input_tensor_0], input_0)
        self.assertEqual(feed_dict[input_tensor_1], input_1)


if __name__ == "__main__":
    unittest.main()
