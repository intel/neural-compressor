#
#  -*- coding: utf-8 -*-
#
import os
import platform
import unittest

import tensorflow as tf

import neural_compressor
from neural_compressor.tensorflow import Model
from neural_compressor.tensorflow.algorithms.static_quant.tensorflow import TensorflowQuery
from neural_compressor.tensorflow.quantization.utils.graph_converter import GraphConverter
from neural_compressor.tensorflow.quantization.utils.utility import read_graph


class TestGraphLibraryDetection(unittest.TestCase):
    efficientnet_b0_model_url = "https://raw.githubusercontent.com/SkyAI/inference_benchmark/435c7ca2577830025ca5f6cbce8480db16f76a61/efficientnet-b0.pb"
    pb_path = "/tmp/.neural_compressor/efficientnet-b0.pb"
    if platform.system().lower() == "windows":
        pb_path = "C:\\tmp\\.neural_compressor\\efficientnet-b0.pb"

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path) and platform.system().lower() == "linux":
            os.system(
                "mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(
                    self.efficientnet_b0_model_url, self.pb_path
                )
            )

    def test_tensorflow_graph_library_detection(self):
        tf.compat.v1.disable_eager_execution()

        op_wise_sequences = TensorflowQuery(
            local_config_file=neural_compressor.__path__[0] + "/tensorflow/algorithms/static_quant/tensorflow.yaml"
        ).get_eightbit_patterns()

        qt_config = {"calib_iteration": 1, "op_wise_config": {}}
        original_graphdef = read_graph(self.pb_path)
        model = Model(self.pb_path)
        model.name = "test"
        model.input_tensor_names = ["input_tensor"]
        model.output_tensor_names = ["softmax_tensor"]
        model.workspace_path = "/tmp/test.pb"

        converter = GraphConverter(model, int8_sequences=op_wise_sequences, qt_config=qt_config)
        converted_graph = converter.convert()

        self.assertEqual(converted_graph.graph_def.library, original_graphdef.library)


if __name__ == "__main__":
    unittest.main()
