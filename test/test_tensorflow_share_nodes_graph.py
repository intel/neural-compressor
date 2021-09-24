#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import copy
from neural_compressor.adaptor.tf_utils.util import read_graph
from neural_compressor.adaptor.tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.split_shared_input import SplitSharedInputOptimizer
class TestTensorflowShareNodesGraphParsing(unittest.TestCase):
    ssd_resnet50_model = 'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz'
    dst_path = '/tmp/.neural_compressor/ssd_resnet50_v1.tgz'
    unzipped_folder_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.dst_path):
          os.system(
              "mkdir -p /tmp/.neural_compressor && wget {} -O {}".format(
                  self.ssd_resnet50_model, self.dst_path, self.dst_path))

        os.system("tar xvf {}".format(self.dst_path))

    @classmethod
    def tearDownClass(self):
        os.system(
            'rm -rf {}'.format(self.unzipped_folder_name))

    def test_parse_pb_contains_share_nodes(self):
        original_graphdef = read_graph(os.path.join(self.unzipped_folder_name, "frozen_inference_graph.pb"))
        copied_graphdef = copy.deepcopy(original_graphdef)
        parsed_graphdef = SplitSharedInputOptimizer(original_graphdef).do_transformation()
        legacy_graphdef = QuantizeGraphHelper.split_shared_inputs(copied_graphdef)
        self.assertGreater(len(parsed_graphdef.node), len(original_graphdef.node))
        self.assertEqual(len(legacy_graphdef.node), len(parsed_graphdef.node))

if __name__ == '__main__':
    unittest.main()
