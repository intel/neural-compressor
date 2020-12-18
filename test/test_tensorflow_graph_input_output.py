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
            exit_policy:
              max_trials: 1
            accuracy_criterion:
              relative: 0.01
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_2.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()
class TestGraphInputOutputDetection(unittest.TestCase):
    rn50_fp32_pb_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb'
    pb_path = '/tmp/mobilenetv1_fp32_pretrained_model.pb'
    inputs = ['input']
    outputs = ['MobilenetV1/Predictions/Reshape_1']

    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        build_fake_yaml_2()
        os.system('wget {} -O {} '.format(self.rn50_fp32_pb_url, self.pb_path))
        self.input_graph = tf.compat.v1.GraphDef()
        with open(self.pb_path, "rb") as f:
            self.input_graph.ParseFromString(f.read())

    @classmethod
    def tearDownClass(self):
        os.system(
            'rm -rf {}'.format(self.pb_path))
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
        from lpot import Quantization

        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(20, 224, 224, 3), label=True)
        dataloader = quantizer.dataloader(dataset, batch_size=2)
        output_graph = quantizer(
            float_graph_def,
            q_dataloader=dataloader,
            eval_dataloader=dataloader
        )
        self.assertGreater(len(output_graph.as_graph_def().node), 0)

    def test_invalid_input_output_config(self):
        g = GraphAnalyzer()
        g.graph = self.input_graph
        g.parse_graph()

        float_graph_def = g.dump_graph()
        from lpot import Quantization

        quantizer = Quantization('fake_yaml_2.yaml')
        dataset = quantizer.dataset('dummy', shape=(20, 224, 224, 3), label=True)
        dataloader = quantizer.dataloader(dataset, batch_size=2)
        catch_assert = False
        try:
          quantizer(
              float_graph_def,
              q_dataloader=dataloader,
              eval_dataloader=dataloader
          )
        except AssertionError as e:
          catch_assert = True
        self.assertEqual(catch_assert, True)

if __name__ == '__main__':
    unittest.main()
