#
#  -*- coding: utf-8 -*-
#
import unittest
import os
from lpot.adaptor.tf_utils.util import get_graph_def
from lpot.adaptor.tf_utils.util import validate_graph_input, validate_graph_output
from lpot.adaptor.tf_utils.util import get_input_node_names, get_output_node_names
from lpot.adaptor.tensorflow import TensorFlowAdaptor


class TestTFAutoDetectInputOutput(unittest.TestCase):
    mb_model_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb'
    pb_path = '~/mobilenet_fp32.pb'

    @classmethod
    def setUpClass(self):
        self.saved_flag = True
        try:
            os.system("wget {} -O {} ".format(self.mb_model_url, self.pb_path))
        except Exception as e:
            self.saved_flag = False

    @classmethod
    def tearDownClass(self):
        if not self.saved_flag:
            os.system("rm -rf {}".format(self.pb_path))

    def testAutoDetectInputOutput(self):
        if self.saved_flag:
            graph_def = get_graph_def(self.pb_path)
            outputs = get_output_node_names(graph_def)
            inputs = get_input_node_names(graph_def)
            output_validate = validate_graph_output(graph_def, outputs)
            self.assertTrue(output_validate)

            input_validate = validate_graph_input(graph_def, inputs)
            self.assertTrue(input_validate)
            framework_specific_info = {'device': 'cpu', 'workspace_path': './'}
            adaptor = TensorFlowAdaptor(framework_specific_info)  
            adaptor._validate_and_inference_input_output(graph_def)
            self.assertTrue(len(adaptor.input_node_names) > 0)
            self.assertTrue(len(adaptor.output_node_names) > 0)

if __name__ == "__main__":
    unittest.main()
