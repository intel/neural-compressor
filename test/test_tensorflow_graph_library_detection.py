#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import tensorflow as tf

from lpot.adaptor.tf_utils.util import read_graph
from lpot.adaptor.tf_utils.graph_converter import GraphConverter
from lpot.adaptor.tensorflow import TensorflowQuery
from lpot.model.model import TensorflowModel
class TestGraphLibraryDetection(unittest.TestCase):
    efficientnet_b0_model_url = 'https://raw.githubusercontent.com/SkyAI/inference_benchmark/435c7ca2577830025ca5f6cbce8480db16f76a61/efficientnet-b0.pb'
    pb_path = '/tmp/.lpot/efficientnet-b0.pb'

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path):
            os.system("mkdir -p /tmp/.lpot && wget {} -O {} ".format(self.efficientnet_b0_model_url, self.pb_path))

    def test_tensorflow_graph_library_detection(self):

        tf.compat.v1.disable_eager_execution()

        op_wise_sequences = TensorflowQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "../lpot/adaptor/tensorflow.yaml")).get_eightbit_patterns()

        qt_config = {'calib_iteration':1, 'op_wise_config':{}}
        original_graphdef = read_graph(self.pb_path)
        framework_info = {
            'name': 'test',
            'input_tensor_names': 'input_tensor',
            'output_tensor_names': 'softmax_tensor',
            'workspace_path': "/tmp/test.pb"
        }
        model = TensorflowModel(self.pb_path, framework_info)
        converter = GraphConverter(model,
                                   int8_sequences=op_wise_sequences,
                                   qt_config=qt_config
                                   )
        converted_graph = converter.convert()

        self.assertEqual(converted_graph.graph_def.library, original_graphdef.library)

if __name__ == "__main__":
    unittest.main()
