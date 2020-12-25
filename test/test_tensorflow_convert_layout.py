import unittest
import tensorflow as tf
from tensorflow.python.framework import graph_util

class TestConvertLayout(unittest.TestCase):
    def test_convert_layout(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, [1, 10, 10, 3], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 10, 3],
                            initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="VALID", 
                                data_format='NCHW')
        relu = tf.nn.relu(conv, name='relu')
        out_name = relu.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

        from lpot.adaptor.tf_utils.graph_rewriter.generic import convert_layout
        convert = convert_layout.ConvertLayoutOptimizer(output_graph_def, [out_name])
        if tf.version.VERSION >= '2.4.0':
            convert_graph = convert.do_transformation()
            for node in convert_graph.node:
                if node.op == 'Conv2D' and 'data_format' in node.attr:
                    self.assertEqual(node.attr['data_format'].s, b'NHWC')
        else:
            self.assertRaises(AssertionError, convert.do_transformation)

if __name__ == "__main__":
    unittest.main()
