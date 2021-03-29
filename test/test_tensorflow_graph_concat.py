#
#
#  -*- coding: utf-8 -*-
import unittest
import os
import tensorflow as tf
import yaml

from lpot.adaptor.tf_utils.util import read_graph
from lpot.adaptor.tf_utils.quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from lpot.adaptor.tensorflow import TensorflowQuery


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: predict
        device: cpu
        quantization:
          model_wise:
            weight:
                granularity: per_tensor
                scheme: sym
                dtype: int8
                algorithm: kl
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: mse
            accuracy_criterion:
              relative: 0.01
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

class TestTensorflowConcat(unittest.TestCase):
    mb_model_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv3_fp32_pretrained_model.pb'
    pb_path = '/tmp/.lpot/inceptionv3_fp32.pb'

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path):
            os.system("mkdir -p /tmp/.lpot && wget {} -O {} ".format(self.mb_model_url, self.pb_path))
        self.op_wise_sequences = TensorflowQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "../lpot/adaptor/tensorflow.yaml")).get_eightbit_patterns()
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
    
    def test_tensorflow_concat_quantization(self):

        output_graph_def = read_graph(self.pb_path)

        from lpot.experimental import Quantization, common
        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 299, 299, 3), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = output_graph_def
        output_graph = quantizer()
        found_quantized_concat_node = False

        target_concat_node_name = 'v0/cg/incept_v3_a0/concat_eightbit_quantized_concatv2'
        from lpot.adaptor.tf_utils.graph_rewriter.graph_util import GraphAnalyzer 
        cur_graph = GraphAnalyzer()
        cur_graph.graph = output_graph.graph_def
        graph_info = cur_graph.parse_graph()
        found_quantized_concat_node = target_concat_node_name in graph_info

        self.assertEqual(found_quantized_concat_node, True)
        min_out, max_out = [], []
        for input_conv_name in graph_info[target_concat_node_name].node.input[:4]:
            # print (input_conv_name, graph_info[input_conv_name].node.input)
            min_freezed_out_name = graph_info[input_conv_name].node.input[-2]
            max_freezed_out_name = graph_info[input_conv_name].node.input[-1]
            min_freezed_out_value = (graph_info[min_freezed_out_name].node.attr['value'].tensor.float_val)[0]
            max_freezed_out_value = (graph_info[max_freezed_out_name].node.attr['value'].tensor.float_val)[0]
            min_out.append(min_freezed_out_value) 
            max_out.append(max_freezed_out_value)
        
        self.assertEqual(len(set(min_out)), 1)
        self.assertEqual(len(set(max_out)), 1)

if __name__ == "__main__":
    unittest.main()
