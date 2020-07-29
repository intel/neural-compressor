#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import tensorflow as tf
from ilit.adaptor.tf_utils.util import parse_ckpt_model, is_ckpt_format


class TestCkptConvert(unittest.TestCase):
    mobilenet_ckpt_url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz'
    dst_path = '/tmp/mobilenet_v1_1.0_224.tgz'

    @classmethod
    def setUpClass(self):
        os.system(
            "wget {} -O {} && mkdir -p ckpt && tar xvf {} -C ckpt".format(
                self.mobilenet_ckpt_url, self.dst_path, self.dst_path))

    @classmethod
    def tearDownClass(self):
        os.system('rm -rf ckpt')

    def test_detect_ckpt_model(self):
        ckpt_prefix = is_ckpt_format('ckpt')
        self.assertNotEqual(ckpt_prefix, None)

    def test_convert_ckpt(self):
        output_names = ['MobilenetV1/Predictions/Reshape_1']
        ckpt_prefix = is_ckpt_format('ckpt')

        tf.compat.v1.disable_eager_execution()
        converted_graph_def = parse_ckpt_model('ckpt/' + ckpt_prefix,
                                               output_names)
        self.assertNotEqual(converted_graph_def, None)


if __name__ == "__main__":
    unittest.main()
