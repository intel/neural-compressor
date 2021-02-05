import unittest
import numpy as np

import tensorflow.compat.v1 as tf
from lpot.adaptor.tf_utils.graph_rewriter.generic.grappler_pass import GrapplerOptimizer
from lpot.adaptor.tf_utils.util import disable_random


class TestGrapplerPass(unittest.TestCase):
    @disable_random()
    def test_grappler_pass(self):

        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float)
            z_data = np.array([[2, 4], [6, 8]], dtype=np.float)
            x = tf.placeholder(tf.float32, shape=[2, 2], name='x')
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.constant(z_data, dtype=tf.float32, shape=[2, 2])
            y1 = tf.math.add(y, z)
            y2 = tf.math.add(y1, z)
            z = tf.matmul(x, y2)
            z = tf.nn.bias_add(z, [1, 2])
            p = tf.identity(z)
            z = tf.identity(p, name='op_to_store')
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                float_graph_def = sess.graph.as_graph_def()
                optimized_graph = GrapplerOptimizer(
                    float_graph_def, ['op_to_store']).do_transformation()
                identity_count = 0
                for i in optimized_graph.node:
                    if i.op == 'Identity':
                        identity_count += 1

                self.assertEqual(identity_count, 1)


if __name__ == "__main__":
    unittest.main()
