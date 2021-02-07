#
#  -*- coding: utf-8 -*-
#
import unittest
import os
from lpot.adaptor.tensorflow import TensorFlowAdaptor
from lpot.model.model import TensorflowModel, validate_graph_node


class TestTFAutoDetectInputOutput(unittest.TestCase):
    mb_model_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb'
    pb_path = '/tmp/.lpot/mobilenet_fp32.pb'

    @classmethod
    def setUpClass(self):
        self.saved_flag = True
        if not os.path.exists(self.pb_path):
            try:
                os.system("mkdir -p /tmp/.lpot && wget {} -O {} ".format(self.mb_model_url, self.pb_path))
            except Exception as e:
                self.saved_flag = False

    def testAutoDetectInputOutput(self):
        if self.saved_flag:
            model = TensorflowModel(self.pb_path)
            outputs = model.output_node_names
            inputs = model.input_node_names
            output_validate = validate_graph_node(model.graph_def, outputs)
            self.assertTrue(output_validate)

            input_validate = validate_graph_node(model.graph_def, inputs)
            self.assertTrue(input_validate)

if __name__ == "__main__":
    unittest.main()
