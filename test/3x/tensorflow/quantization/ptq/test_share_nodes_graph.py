#
#  -*- coding: utf-8 -*-
#
import copy
import os
import platform
import tarfile
import unittest

from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.split_shared_input import (
    SplitSharedInputOptimizer,
)
from neural_compressor.tensorflow.quantization.utils.quantize_graph_common import QuantizeGraphHelper
from neural_compressor.tensorflow.quantization.utils.utility import read_graph


class TestTensorflowShareNodesGraphParsing(unittest.TestCase):
    ssd_resnet50_model = "http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz"
    dst_path = "/tmp/.neural_compressor/ssd_resnet50_v1.tgz"
    platform = platform.system().lower()
    if platform == "windows":
        unzipped_folder_name = "C:\\tmp\\.neural_compressor\ssd_resnet50_v1\\ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
    else:
        unzipped_folder_name = "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"

    @classmethod
    def setUpClass(self):
        if self.platform == "linux":
            if not os.path.exists(self.dst_path):
                os.system(
                    "mkdir -p /tmp/.neural_compressor && wget {} -O {}".format(
                        self.ssd_resnet50_model,
                        self.dst_path,
                    )
                )
            os.system("tar xvf {}".format(self.dst_path))
        elif self.platform == "windows":
            if not os.path.exists(self.unzipped_folder_name):
                os.system("md C:\\tmp\.neural_compressor && cd C:\\tmp\.neural_compressor")
                from urllib import request

                request.urlretrieve(self.ssd_resnet50_model, self.dst_path)
                tar = tarfile.open(self.dst_path)
                tar.extractall(self.unzipped_folder_name)

    @classmethod
    def tearDownClass(self):
        if self.platform == "linux":
            os.system("rm -rf {}".format(self.unzipped_folder_name))

    def test_parse_pb_contains_share_nodes(self):
        original_graphdef = read_graph(os.path.join(self.unzipped_folder_name, "frozen_inference_graph.pb"))
        copied_graphdef = copy.deepcopy(original_graphdef)
        parsed_graphdef = SplitSharedInputOptimizer(original_graphdef).do_transformation()
        legacy_graphdef = QuantizeGraphHelper.split_shared_inputs(copied_graphdef)
        self.assertGreater(len(parsed_graphdef.node), len(original_graphdef.node))
        self.assertEqual(len(legacy_graphdef.node), len(parsed_graphdef.node))


if __name__ == "__main__":
    unittest.main()
