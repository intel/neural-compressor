#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import tensorflow as tf
from lpot.adaptor.tf_utils.util import get_slim_graph

class TestSlimCkptConvert(unittest.TestCase):
    inception_ckpt_url = 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'
    dst_path = '/tmp/inception_v1_2016_08_28.tar.gz'

    @classmethod
    def setUpClass(self):
        os.system(
            "wget {} -O {} && mkdir -p ckpt && tar xvf {} -C ckpt".format(
                self.inception_ckpt_url, self.dst_path, self.dst_path))

    @classmethod
    def tearDownClass(self):
        os.system('rm -rf ckpt')

    def test_get_slim_graph(self):
        # only support tensorflow1.x
        if tf.version.VERSION > '2.0.0':
            return
        from tf_slim.nets import inception  
        model_func = inception.inception_v1
        arg_scope = inception.inception_v1_arg_scope()
        kwargs = {'num_classes': 1001}
        inputs_shape = [None, 224, 224, 3]
        images = tf.compat.v1.placeholder(name='input', \
            dtype=tf.float32, shape=inputs_shape)
        graph = get_slim_graph('./ckpt/inception_v1.ckpt', model_func, \
            arg_scope, images, **kwargs)
        self.assertTrue(isinstance(graph, tf.Graph)) 
        graph_def = graph.as_graph_def()
        self.assertGreater(len(graph_def.node), 1)

if __name__ == "__main__":
    unittest.main()
