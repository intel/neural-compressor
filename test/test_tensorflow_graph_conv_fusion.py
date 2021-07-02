#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import yaml
import numpy as np
import tensorflow as tf

from lpot.adaptor.tf_utils.quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from lpot.adaptor.tf_utils.graph_rewriter.generic.strip_unused_nodes import StripUnusedNodesOptimizer
from lpot.adaptor.tf_utils.graph_rewriter.generic.fold_batch_norm import FoldBatchNormNodesOptimizer
from tensorflow.python.framework import graph_util
from lpot.adaptor.tensorflow import TensorflowQuery
from lpot.adaptor.tf_utils.util import disable_random


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
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
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


class TestConvBiasAddAddReluFusion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @disable_random()
    def test_conv_relu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        relu = tf.nn.relu(conv)

        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()
            found_conv_fusion = True

            for i in output_graph.graph_def.node:
                if i.op == 'Relu':
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, False)

    @unittest.skipUnless(bool(
            tf.version.VERSION.find('1.15.0-up') != -1 or tf.version.VERSION >= '2.1.0'), 'not supported the current tf version.')
    @disable_random()
    def test_conv_biasadd_relu6_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu6 = tf.nn.relu6(normed, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()
            found_conv_fusion = True

            for i in output_graph.graph_def.node:
                if i.op == 'Relu6':
                    found_conv_fusion = False
                    break
            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_biasadd_addv2_relu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        # relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        # relu2 = tf.nn.relu(normed2)
        add = tf.raw_ops.AddV2(x=normed, y=normed2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            from lpot.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            found_conv_fusion = False

            for i in output_graph.graph_def.node:
                if i.op == 'QuantizedConv2DWithBiasSignedSumAndReluAndRequantize':
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_fusion_with_last_matmul(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        # paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        # x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(top_relu, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        pooling = tf.nn.max_pool(relu, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        reshape = tf.reshape(pooling, [-1, 3136])

        y_data = np.random.random([3136, 1])

        y = tf.constant(y_data, dtype=tf.float32, shape=[3136, 1])
        z = tf.matmul(reshape, y)
        relu1 = tf.nn.relu(z)
        y_data_1 = np.random.random([1, 1])
        y_1 = tf.constant(y_data_1, dtype=tf.float32, shape=[1, 1])

        z_2nd_matmul = tf.matmul(relu1, y_1)
        relu6 = tf.nn.relu6(z_2nd_matmul, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            from lpot.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            quantize_v2_count = 0
            for i in output_graph.graph_def.node:
                if i.op == 'QuantizeV2':
                    quantize_v2_count += 1
                    break

            self.assertEqual(quantize_v2_count, 1)

    @disable_random()
    def test_conv_fusion_with_last_conv(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(top_relu, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed)
        pooling = tf.nn.max_pool(relu, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        conv_weights_2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                   initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(pooling, conv_weights_2, strides=[1, 2, 2, 1], padding="VALID")
        conv_weights_3 = tf.compat.v1.get_variable("weight3", [3, 3, 16, 16],
                                                   initializer=tf.compat.v1.random_normal_initializer())
        relu2 = tf.nn.relu(conv2)
        conv3 = tf.nn.conv2d(relu2, conv_weights_3, strides=[1, 2, 2, 1], padding="VALID")

        relu3 = tf.nn.relu(conv3)
        relu6 = tf.nn.relu6(relu3, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            from lpot.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            quantize_v2_count = 0
            for i in output_graph.graph_def.node:
                if i.op == 'QuantizeV2':
                    quantize_v2_count += 1
                    break

            self.assertEqual(quantize_v2_count, 1)

    @disable_random()
    def test_conv_fusion_with_max_pooling(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")

        relu = tf.nn.relu(x)
        pooling = tf.nn.max_pool(relu, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        conv_weights = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                   initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(pooling, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        biasadd = tf.compat.v1.layers.batch_normalization(conv, name='op_to_store')
        out_name = biasadd.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            from lpot.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer()

            quantized_pool_data_type = None
            quantized_conv_data_type = None
            for i in output_graph.graph_def.node:
                if i.op.find("QuantizedMaxPool") != -1:
                    quantized_pool_data_type = i.attr['T'].type
                if i.op.find("QuantizedConv2D") != -1:
                    quantized_conv_data_type = i.attr['Tinput'].type

            self.assertNotEqual(quantized_pool_data_type, None)
            self.assertEqual(quantized_pool_data_type, quantized_conv_data_type)
class TestGraphConvFusion(unittest.TestCase):
    rn50_fp32_pb_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb'
    pb_path = '/tmp/.lpot/resnet50_fp32_pretrained_model.pb'
    inputs = ['input']
    outputs = ['predict']

    op_wise_config = {
        "v0/resnet_v13/conv14/conv2d/Conv2D": (False, 'minmax', False, 7.0),
        "v0/resnet_v13/conv11/conv2d/Conv2D": (False, 'minmax', False, 7.0),
        "v0/resnet_v17/conv27/conv2d/Conv2D": (False, 'minmax', False, 7.0)
    }

    @classmethod
    def setUpClass(self):
        if not os.path.exists(self.pb_path):
            os.system('mkdir -p /tmp/.lpot && wget {} -O {} '.format(self.rn50_fp32_pb_url, self.pb_path))
        self.input_graph = tf.compat.v1.GraphDef()
        with open(self.pb_path, "rb") as f:
            self.input_graph.ParseFromString(f.read())

    def test_conv_biasadd_relu_fusion(self):
        tf.compat.v1.disable_eager_execution()

        self._tmp_graph_def = graph_util.remove_training_nodes(self.input_graph, self.outputs)

        self._tmp_graph_def = StripUnusedNodesOptimizer(self._tmp_graph_def,
                                                        self.inputs, self.outputs).do_transformation()

        self._tmp_graph_def = FoldBatchNormNodesOptimizer(self._tmp_graph_def).do_transformation()
        op_wise_sequences = TensorflowQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "../lpot/adaptor/tensorflow.yaml")).get_eightbit_patterns()

        output_graph, _ = QuantizeGraphForIntel(self._tmp_graph_def, self.outputs,
                                             self.op_wise_config, op_wise_sequences,
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
