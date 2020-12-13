#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import tensorflow as tf
from lpot.adaptor.tf_utils.util import parse_savedmodel_model, is_saved_model_format, get_graph_def


class TestSavedModelToPbConvert(unittest.TestCase):
    mobilenet_ckpt_url = 'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz'
    dst_path = '/tmp/ssd_resnet50_v1.tgz'

    @classmethod
    def setUpClass(self):
        os.system(
            "wget {} -O {} && tar xvf {}".format(
                self.mobilenet_ckpt_url, self.dst_path, self.dst_path))

    @classmethod
    def tearDownClass(self):
        os.system(
            'rm -rf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03')

    def test_detect_savedmodel_model(self):
        res = is_saved_model_format(
            'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/saved_model')
        self.assertEqual(res, True)

    def test_convert_savedmodel(self):
        tf.compat.v1.disable_eager_execution()
        converted_graph_def, _, _ = parse_savedmodel_model(
            'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/saved_model')
        self.assertNotEqual(converted_graph_def, None)


if __name__ == "__main__":
    unittest.main()
