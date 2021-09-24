#
#
#  -*- coding: utf-8 -*-
import unittest
import os
import tensorflow as tf
import yaml

from neural_compressor.adaptor.tf_utils.util import read_graph
from neural_compressor.adaptor.tf_utils.quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from neural_compressor.adaptor.tensorflow import TensorflowQuery
from neural_compressor.adaptor.tf_utils.util import disable_random
from tensorflow.python.framework import graph_util


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
                algorithm: minmax
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
    pb_path = '/tmp/.neural_compressor/inceptionv3_fp32.pb'

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path):
            os.system(
                "mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(self.mb_model_url, self.pb_path))
        self.op_wise_sequences = TensorflowQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "../neural_compressor/adaptor/tensorflow.yaml")).get_eightbit_patterns()
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    def test_tensorflow_concat_quantization(self):

        output_graph_def = read_graph(self.pb_path)

        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset(
            'dummy', shape=(100, 299, 299, 3), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = output_graph_def
        output_graph = quantizer()
        found_quantized_concat_node = False

        target_concat_node_name = 'v0/cg/incept_v3_a0/concat_eightbit_quantized_concatv2'
        from neural_compressor.adaptor.tf_utils.graph_rewriter.graph_util import GraphAnalyzer
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
            min_freezed_out_value = (
                graph_info[min_freezed_out_name].node.attr['value'].tensor.float_val)[0]
            max_freezed_out_value = (
                graph_info[max_freezed_out_name].node.attr['value'].tensor.float_val)[0]
            min_out.append(min_freezed_out_value)
            max_out.append(max_freezed_out_value)

        self.assertEqual(len(set(min_out)), 1)
        self.assertEqual(len(set(max_out)), 1)

    @disable_random()
    def test_concat_with_different_input_type(self):
        x = tf.compat.v1.placeholder(
            tf.float32, [1, 128, 128, 16], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [2, 2, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [16],
                                              initializer=tf.compat.v1.random_normal_initializer())

        x = tf.nn.relu(x)
        sqrt = tf.math.sqrt(x)
        relu_sqrt = tf.nn.relu(sqrt)
        conv = tf.nn.conv2d(relu_sqrt, conv_weights, strides=[
            1, 2, 2, 1], padding="SAME", name='last')
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[
            1, 2, 2, 1], padding="SAME", name='last')
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        concat = tf.concat([relu, conv_bias], 1)
        final_node = tf.nn.relu(concat, name='op_to_store')
        out_name = final_node.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset(
                'dummy', shape=(100, 128, 128, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            quantized_concat = False
            for i in output_graph.graph_def.node:
              if i.op == 'QuantizedConcatV2':
                  quantized_concat = True
            self.assertEqual(quantized_concat, False)


if __name__ == "__main__":
    unittest.main()
