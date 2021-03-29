#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import yaml
import tensorflow as tf

from lpot.adaptor.tf_utils.graph_rewriter.graph_util import GraphAnalyzer


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
        device: cpu
        quantization:
          model_wise:
            weight:
                granularity: per_tensor
                scheme: sym
                dtype: int8
                algorithm: minmax
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: basic
            accuracy_criterion:
              relative: 0.1
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

def build_fake_yaml_2():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: x
          outputs: op_to_store
        device: cpu
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: bayesian
            accuracy_criterion:
              relative: 0.01
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_2.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()
class TestGraphInputOutputDetection(unittest.TestCase):
    mb_fp32_pb_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb'
    pb_path = '/tmp/.lpot/mobilenet_fp32.pb'
    inputs = ['input']
    outputs = ['MobilenetV1/Predictions/Reshape_1']

    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        build_fake_yaml_2()
        if not os.path.exists(self.pb_path):
            os.system('mkdir -p /tmp/.lpot && wget {} -O {} '.format(self.mb_fp32_pb_url, self.pb_path))
        self.input_graph = tf.compat.v1.GraphDef()
        with open(self.pb_path, "rb") as f:
            self.input_graph.ParseFromString(f.read())

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('fake_yaml_2.yaml')

    def test_identify_input_output(self):
        g = GraphAnalyzer()
        g.graph = self.input_graph
        g.parse_graph()
        inputs, outputs = g.get_graph_input_output()
        self.assertEqual(inputs, self.inputs)
        self.assertEqual(outputs, self.outputs)

    def test_no_input_output_config(self):
        g = GraphAnalyzer()
        g.graph = self.input_graph
        g.parse_graph()

        float_graph_def = g.dump_graph()
        from lpot.experimental import Quantization, common

        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(20, 224, 224, 3), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=2)
        quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=2)
        quantizer.model = float_graph_def
        output_graph = quantizer()
        self.assertGreater(len(output_graph.graph_def.node), 0)

    def test_invalid_input_output_config(self):
        g = GraphAnalyzer()
        g.graph = self.input_graph
        g.parse_graph()

        float_graph_def = g.dump_graph()
        from lpot.experimental import Quantization, common

        quantizer = Quantization('fake_yaml_2.yaml')
        dataset = quantizer.dataset('dummy', shape=(20, 224, 224, 3), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=2)
        quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=2)
        quantizer.model = float_graph_def
        model = quantizer()
        # will detect the right inputs/outputs
        self.assertNotEqual(model.input_node_names, ['x'])
        self.assertNotEqual(model.output_node_names, ['op_to_store'])

if __name__ == '__main__':
    unittest.main()
