#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import tensorflow as tf

from neural_compressor.adaptor.tf_utils.util import read_graph
from neural_compressor.adaptor.tf_utils.quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from neural_compressor.adaptor.tensorflow import TensorflowQuery
class TestTensorflowGpu(unittest.TestCase):
    mb_model_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb'
    pb_path = '/tmp/.neural_compressor/mobilenet_fp32.pb'

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path):
            os.system("mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(self.mb_model_url, self.pb_path))
        self.op_wise_sequences = TensorflowQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "../neural_compressor/adaptor/tensorflow.yaml")).get_eightbit_patterns()

    def test_tensorflow_gpu_conversion(self):
        input_graph_def = read_graph(self.pb_path)
        output_node_names = ['MobilenetV1/Predictions/Reshape_1']
        op_wise_config = {
            'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D': (False, 'minmax', False, 7.0)}
        tf.compat.v1.disable_eager_execution()

        converter = QuantizeGraphForIntel(
            input_graph_def, output_node_names, op_wise_config, self.op_wise_sequences,  'gpu')
        converted_pb, _ = converter.do_transform()

        target_node_name = 'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D_eightbit_quantized_conv'

        node_details = {}
        for i in converted_pb.node:
            node_details[i.name] = i

        converted_flag = True if target_node_name in node_details else False

        self.assertEqual(converted_flag, True)

        target_node = node_details[target_node_name]
        weights_min_node = node_details[target_node.input[-2]]
        weights_max_node = node_details[target_node.input[-1]]

        self.assertEqual(weights_max_node.op, "HostConst")
        self.assertEqual(weights_min_node.op, "HostConst")

if __name__ == "__main__":
    unittest.main()
