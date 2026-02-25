#
#  -*- coding: utf-8 -*-
#
import os
import platform
import unittest

from neural_compressor.tensorflow import Model as TensorflowModel
from neural_compressor.tensorflow.algorithms.static_quant.tensorflow import TensorFlowAdaptor
from neural_compressor.tensorflow.utils.model_wrappers import validate_graph_node


class TestTFAutoDetectInputOutput(unittest.TestCase):
    mb_model_url = (
        "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb"
    )
    pb_path = "/tmp/.neural_compressor/mobilenet_fp32.pb"
    platform = platform.system().lower()
    if platform == "windows":
        pb_path = "C:\\tmp\\.neural_compressor\\mobilenet_fp32.pb"

    @classmethod
    def setUpClass(self):
        self.saved_flag = True
        if not os.path.exists(self.pb_path):
            try:
                if self.platform == "linux":
                    os.system(
                        "mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(self.mb_model_url, self.pb_path)
                    )
                elif self.platform == "windows":
                    os.system("md C:\\tmp\.neural_compressor && cd C:\\tmp\.neural_compressor")
                    from urllib import request

                    request.urlretrieve(self.mb_model_url)
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
