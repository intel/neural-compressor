#
#  -*- coding: utf-8 -*-
#
import unittest
import yaml
import os
import tensorflow as tf


from neural_compressor.adaptor.tensorflow import TensorflowQuery
from neural_compressor.adaptor.tf_utils.util import disable_random
from tensorflow.python.framework import graph_util


def build_fake_yaml_on_grappler():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          recipes:
            scale_propagation_concat: False
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
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_grappler.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

class TestTFQueryYaml(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tf_yaml_path = os.path.join(os.getcwd() + "/../neural_compressor/adaptor/tensorflow.yaml")

        with open(self.tf_yaml_path) as f:
            self.content = yaml.safe_load(f)
        self.query_handler = TensorflowQuery(local_config_file=self.tf_yaml_path)
        build_fake_yaml_on_grappler()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml_grappler.yaml')

    def test_unique_version(self):
        versions = [i['version']['name'] for i in self.content]
        registered_version_name = []
        for i in versions:
          if isinstance(i, list):
            registered_version_name.extend(i)
          else:
            registered_version_name.append(i)

        self.assertEqual(len(registered_version_name), len(set(registered_version_name)))

    def test_int8_sequences(self):
        patterns = self.query_handler.get_eightbit_patterns()

        has_conv2d = bool('Conv2D' in patterns)
        has_matmul = bool('MatMul' in patterns)
        self.assertEqual(has_conv2d, True)
        self.assertEqual(has_matmul, True)
        self.assertGreaterEqual(len(patterns['Conv2D']), 13)
        self.assertGreaterEqual(len(patterns['MatMul']), 3)
        self.assertEqual(len(patterns['ConcatV2']), 1)
        self.assertEqual(len(patterns['MaxPool']), 1)
        self.assertEqual(len(patterns['AvgPool']), 1)

    def test_convert_internal_patterns(self):
        internal_patterns = self.query_handler.generate_internal_patterns()
        self.assertEqual([['MaxPool']] in internal_patterns, True)
        self.assertEqual([['ConcatV2']] in internal_patterns, True)
        self.assertEqual([['AvgPool']] in internal_patterns, True)
        self.assertEqual([['MatMul'], ('BiasAdd',), ('Relu',)] in internal_patterns, True)

    @disable_random()
    def test_grappler_cfg(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 30, 30, 1], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [2, 2, 1, 1],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [1],
                                              initializer=tf.compat.v1.random_normal_initializer())

        x = tf.nn.relu(x)
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        relu2 = tf.nn.relu(relu)
        pool = tf.nn.max_pool(relu2, ksize=1, strides=[1, 2, 2, 1], name='maxpool', padding="SAME")
        conv1 = tf.nn.conv2d(pool, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        x = tf.nn.relu(conv_bias)
        final_node = tf.nn.relu(x, name='op_to_store')

        out_name = final_node.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization('fake_yaml_grappler.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 30, 30, 1), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            disable_arithmetic = False
            for i in output_graph.graph_def.node:
              if i.name == 'maxpool_eightbit_quantize_Relu_2' and i.input[0] == 'Relu_2':
                disable_arithmetic = True
            # if tf.version.VERSION >= '2.3.0':
            #     self.assertEqual(False, disable_arithmetic)
            # else:
            self.assertEqual(True, disable_arithmetic)

if __name__ == '__main__':
    unittest.main()
