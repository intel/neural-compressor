#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import tensorflow as tf

from ilit.adaptor.tf_utils.quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from ilit.adaptor.tf_utils.transform_graph.strip_unused import StripUnusedNodes
from ilit.adaptor.tf_utils.transform_graph.fold_batch_norm import FoldBatchNormNodes
from tensorflow.python.framework import graph_util

class TestGraphConvFusion(unittest.TestCase):
    rn50_fp32_pb_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb'
    pb_path = '/tmp/resnet50_fp32_pretrained_model.pb'
    inputs = ['input']
    outputs = ['predict']

    op_wise_config= {
        "v0/resnet_v13/conv14/conv2d/Conv2D": (False, 'minmax', False),
        "v0/resnet_v13/conv11/conv2d/Conv2D": (False, 'minmax', False),
        "v0/resnet_v17/conv27/conv2d/Conv2D": (False, 'minmax', False)
    }

    @classmethod
    def setUpClass(self):
        os.system('wget {} -O {} '.format(self.rn50_fp32_pb_url, self.pb_path))
        self.input_graph = tf.compat.v1.GraphDef()
        with open(self.pb_path, "rb") as f:
            self.input_graph.ParseFromString(f.read())

    @classmethod
    def tearDownClass(self):
        os.system(
            'rm -rf {}'.format(self.pb_path))

    def test_conv_biasadd_relu_fusion(self):
        tf.compat.v1.disable_eager_execution()

        self._tmp_graph_def = graph_util.remove_training_nodes(self.input_graph, self.outputs)

        self._tmp_graph_def = StripUnusedNodes(self._tmp_graph_def,
                                                self.inputs, self.outputs).do_transform()

        self._tmp_graph_def = FoldBatchNormNodes(self._tmp_graph_def).do_transform()

        output_graph = QuantizeGraphForIntel(self._tmp_graph_def, self.outputs,
                                             self.op_wise_config,
                                             'cpu').do_transform()

        node_name_type_mapping = {}
        for i in output_graph.node:
            node_name_type_mapping[i.name] = i.op

        should_disable_sum_node_name = 'v0/resnet_v17/conv27/conv2d/Conv2D_eightbit_quantized_conv'
        should_enable_sum_node_name = 'v0/resnet_v13/conv11/conv2d/Conv2D_eightbit_quantized_conv'
        should_disable_sum_flag = should_disable_sum_node_name in node_name_type_mapping and node_name_type_mapping[
            should_disable_sum_node_name] == 'QuantizedConv2DWithBias'
        should_enable_sum_flag = should_enable_sum_node_name in node_name_type_mapping and node_name_type_mapping[
            should_enable_sum_node_name] == 'QuantizedConv2DWithBiasSumAndRelu'
        self.assertEqual(should_enable_sum_flag, True)
        self.assertEqual(should_disable_sum_flag, True)

if __name__ == '__main__':
    unittest.main()
