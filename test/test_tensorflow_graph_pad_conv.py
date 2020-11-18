
import unittest
import tensorflow as tf
from tensorflow.python.framework import graph_util
from ilit.adaptor.tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper
from ilit.adaptor.tf_utils.graph_rewriter.generic.fold_batch_norm import FoldBatchNormNodesOptimizer
from ilit.adaptor.tf_utils.graph_rewriter.generic.fuse_pad_with_conv import FusePadWithConv2DOptimizer
from ilit.adaptor.tf_utils.quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel


class TestFoldPadConv(unittest.TestCase):
    def test_fold_pad_conv(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed)
        out_name = relu.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            output_graph_def = QuantizeGraphHelper.remove_training_nodes(
                output_graph_def, protected_nodes=[out_name])
            output_graph_def = FoldBatchNormNodesOptimizer(output_graph_def).do_transformation()
            outputs = [out_name]
            op_wise_config = {
                "Conv2D": (False, 'minmax', False),
            }
            output_graph_def = FusePadWithConv2DOptimizer(
                output_graph_def, [], ['input'], op_wise_config).do_transformation()
            fold_graph_def = QuantizeGraphForIntel(
                output_graph_def, outputs, op_wise_config, 'cpu').do_transform()
        found_QuantizedConv2DWithBiasAndRelu = False
        found_pad = False
        for i in fold_graph_def.node:
            if i.op == 'Pad':
                found_pad = True

            if i.op == 'QuantizedConv2DWithBiasAndRelu':
                found_QuantizedConv2DWithBiasAndRelu = True

        self.assertEqual(found_QuantizedConv2DWithBiasAndRelu, True)
        self.assertEqual(found_pad, False)

    def test_fold_pad_conv2(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed)

        paddings2 = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad2 = tf.pad(x, paddings2, "CONSTANT")
        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(x_pad2, conv_weights2, strides=[1, 2, 2, 1], padding="VALID")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        relu2 = tf.nn.relu(normed2)
        add = tf.math.add(relu, relu2)
        out_name = add.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            output_graph_def = QuantizeGraphHelper.remove_training_nodes(
                output_graph_def, protected_nodes=[out_name])
            output_graph_def = FoldBatchNormNodesOptimizer(output_graph_def).do_transformation()

            outputs = [out_name]
            op_wise_config = {
                "Conv2D": (False, 'minmax', False),
                "Conv2D_1": (False, 'minmax', False)
            }
            output_graph_def = FusePadWithConv2DOptimizer(
                output_graph_def, [], ['input'], op_wise_config).do_transformation()
            fold_graph_def = QuantizeGraphForIntel(
                output_graph_def, outputs, op_wise_config, 'cpu').do_transform()
        found_QuantizedConv2DWithBiasAndRelu = False
        found_pad = False
        for i in fold_graph_def.node:
            if i.op == "Pad":
                found_pad = True

            if i.op == 'QuantizedConv2DWithBiasAndRelu':
                found_QuantizedConv2DWithBiasAndRelu = True

        self.assertEqual(found_QuantizedConv2DWithBiasAndRelu, True)
        self.assertEqual(found_pad, False)

    def test_fold_pad_conv3(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(x, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        relu2 = tf.nn.relu(normed2)
        add = tf.math.add(relu, relu2)
        out_name = add.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            output_graph_def = QuantizeGraphHelper.remove_training_nodes(
                output_graph_def, protected_nodes=[out_name])
            output_graph_def = FoldBatchNormNodesOptimizer(output_graph_def).do_transformation()
            outputs = [out_name]
            op_wise_config = {
                "Conv2D": (False, 'minmax', False),
                "Conv2D_1": (False, 'minmax', False)
            }
            output_graph_def = FusePadWithConv2DOptimizer(
                output_graph_def, [], ['input'], op_wise_config).do_transformation()
            fold_graph_def = QuantizeGraphForIntel(output_graph_def, outputs,
                                                   op_wise_config,
                                                   'cpu').do_transform()
        found_QuantizedConv2DWithBiasAndRelu = False
        found_pad = False

        for i in fold_graph_def.node:
            if i.op == "Pad":
                found_pad = True

            if i.op == 'QuantizedConv2DWithBiasAndRelu':
                found_QuantizedConv2DWithBiasAndRelu = True

        self.assertEqual(found_QuantizedConv2DWithBiasAndRelu, True)
        self.assertEqual(found_pad, False)

    def test_fold_pad_conv4(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(x, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        relu2 = tf.nn.relu(normed2)
        add = tf.math.add(relu, relu2)
        out_name = add.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            output_graph_def = QuantizeGraphHelper.remove_training_nodes(
                output_graph_def, protected_nodes=[out_name])

            output_graph_def = FoldBatchNormNodesOptimizer(output_graph_def).do_transformation()

            outputs = [out_name]
            op_wise_config = {
                "Pad": (False, 'minmax', False),
                "Conv2D": (False, 'minmax', False),
                "Conv2D_1": (False, 'minmax', False)
            }
            output_graph_def = FusePadWithConv2DOptimizer(
                output_graph_def, [], ['input'], op_wise_config).do_transformation()
            fold_graph_def = QuantizeGraphForIntel(output_graph_def, outputs,
                                                   op_wise_config,
                                                   'cpu').do_transform()
        found_QuantizedConv2DWithBiasAndRelu = False
        found_pad = False

        for i in fold_graph_def.node:
            if i.op == "Pad":
                found_pad = True

            if i.op == 'QuantizedConv2DWithBiasAndRelu':
                found_QuantizedConv2DWithBiasAndRelu = True

        self.assertEqual(found_QuantizedConv2DWithBiasAndRelu, True)
        self.assertEqual(found_pad, True)


if __name__ == "__main__":
    unittest.main()
