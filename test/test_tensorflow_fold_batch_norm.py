import unittest
import tensorflow as tf
import numpy as np
import copy
from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fold_batch_norm import FoldBatchNormNodesOptimizer


class TestFoldBatchnorm(unittest.TestCase):
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")
    conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                             initializer=tf.compat.v1.random_normal_initializer())
    conv_bias = tf.compat.v1.get_variable("bias", [32],
                                          initializer=tf.compat.v1.random_normal_initializer())
    beta = tf.compat.v1.get_variable(name='beta',
                                     shape=[32],
                                     initializer=tf.compat.v1.random_normal_initializer())
    gamma = tf.compat.v1.get_variable(name='gamma',
                                      shape=[32],
                                      initializer=tf.compat.v1.random_normal_initializer())
    conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
    conv_bias = tf.nn.bias_add(conv1, conv_bias)
    normed = tf.compat.v1.layers.batch_normalization(conv_bias)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=[normed.name.split(':')[0]])
        output_graph_def = QuantizeGraphHelper.remove_training_nodes(
            output_graph_def, protected_nodes=[normed.name.split(':')[0]])
        graph_def = copy.deepcopy(output_graph_def)
        fold_graph_def = FoldBatchNormNodesOptimizer(output_graph_def).do_transformation()

    def test_fold_output_values(self):
        input_data = np.random.randn(1, 224, 224, 3)
        graph = tf.compat.v1.Graph()
        fold_graph = tf.compat.v1.Graph()
        with graph.as_default():
            tf.compat.v1.import_graph_def(self.graph_def, name='')

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            x = graph.get_tensor_by_name('input:0')
            normed = graph.get_tensor_by_name('batch_normalization/FusedBatchNormV3:0')
            y = sess.run(normed, feed_dict={x: input_data})

        with fold_graph.as_default():
            tf.compat.v1.import_graph_def(self.fold_graph_def, name='')
        with tf.compat.v1.Session(graph=fold_graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            x = fold_graph.get_tensor_by_name('input:0')
            normed = fold_graph.get_tensor_by_name('batch_normalization/FusedBatchNormV3:0')
            y_fold = sess.run(normed, feed_dict={x: input_data})
        assert np.allclose(y, y_fold, rtol=1e-05, atol=1e-05)

    def test_do_transform(self):
        for node in self.fold_graph_def.node:
            assert node.op not in ["FusedBatchNormV3"]


if __name__ == "__main__":
    unittest.main()
