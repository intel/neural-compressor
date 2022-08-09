#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import yaml
import numpy as np
import tensorflow as tf

from neural_compressor.adaptor.tf_utils.quantize_graph.qdq.optimize_qdq import OptimizeQDQGraph
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.strip_unused_nodes import StripUnusedNodesOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fold_batch_norm import FoldBatchNormNodesOptimizer
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import function
from neural_compressor.adaptor.tensorflow import TensorflowQuery
from neural_compressor.adaptor.tf_utils.util import disable_random
from pkg_resources import parse_version


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: inteltensorflow
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
    def test_conv_single_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv1_weights = tf.compat.v1.get_variable("weight_conv1", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x_pad, conv1_weights, strides=[1, 2, 2, 1], padding="VALID")
        matmul_weights = tf.compat.v1.get_variable("weight_matmul", [1, 28, 16, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        matmul = tf.linalg.matmul(conv1, matmul_weights)
        conv2_weights = tf.compat.v1.get_variable("weight_conv2", [7, 7, 32, 1],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(matmul, conv2_weights, strides=[1, 2, 2, 1], padding="VALID")
        leaky_relu = tf.nn.leaky_relu(conv2, name='op_to_store')

        out_name = leaky_relu.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            find_single_qconv = []
            for i in output_graph.graph_def.node:
                if i.op == '_QuantizedConv2D':
                    find_single_qconv.append(i.attr['fused_ops'].list.s == [b'Requantize'])

            self.assertEqual(find_single_qconv, [True, False])

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
            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()
            found_conv_fusion = True

            for i in output_graph.graph_def.node:
                if i.op == 'Relu':
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, False)

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
            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()
            found_conv_fusion = True

            for i in output_graph.graph_def.node:
                if i.op == 'Relu6':
                    found_conv_fusion = False
                    break
            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_biasadd_swishf32_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        @function.Defun(tf.float32, func_name="swish_f32")
        def swish_f32(x):
            return tf.nn.silu(x, beta=1.0)
        swish = swish_f32(normed, name="swish_f32_output_node")

        out_name = swish.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()
            found_conv_fusion = True

            for i in output_graph.graph_def.node:
                if i.op == 'swish_f32':
                    found_conv_fusion = False
                    break
            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_addv2_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        conv1_weights = tf.compat.v1.get_variable("weight_conv1", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1, 2, 2, 1], padding="SAME")
        conv2_weights = tf.compat.v1.get_variable("weight_conv2", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(x, conv2_weights, strides=[1, 2, 2, 1], padding="SAME")
        sumadd = tf.raw_ops.AddV2(x=conv1, y=conv2, name='addv2')

        out_name = sumadd.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            found_conv_fusion = False
            for i in output_graph.graph_def.node:
                if i.op.find('QuantizedConv2D') != -1:
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_biasadd_add_relu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.nn.bias_add(conv2, tf.constant([3.0, 1.2,1,2,3,4,5,6,7,8,9,0,12,2,3,4]))
        relu = tf.nn.relu(normed2 + tf.constant([3.0]))
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            found_conv_fusion = False

            for i in output_graph.graph_def.node:
                if i.op.find('QuantizedConv2D') != -1:
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_biasadd_addv2_relu_fallback_fusion_1(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.leaky_relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
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

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            found_conv_fusion = False

            for i in output_graph.graph_def.node:
                if i.op == '_QuantizedConv2D' and \
                    i.attr['fused_ops'].list.s == [b'BiasAdd', b'Requantize']:
                    found_conv_fusion = True
                    break
            self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_conv_biasadd_addv2_relu_fallback_fusion_2(self):
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

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            found_conv_fusion = False

            for i in output_graph.graph_def.node:
                if i.op == '_QuantizedConv2D' and \
                    i.attr['fused_ops'].list.s == [b'BiasAdd', b'Requantize']:
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

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

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

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

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

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            quantized_pool_data_type = None
            quantized_conv_data_type = None
            for i in output_graph.graph_def.node:
                if i.op.find("QuantizedMaxPool") != -1:
                    quantized_pool_data_type = i.attr['T'].type
                if i.op.find("QuantizedConv2D") != -1:
                    quantized_conv_data_type = i.attr['Tinput'].type

            self.assertNotEqual(quantized_pool_data_type, None)
            self.assertEqual(quantized_pool_data_type, quantized_conv_data_type)

    @disable_random()
    def test_conv3d_addv2_relu_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 128, 64, 64, 16], name="input")
        top_relu = tf.nn.relu(x)
        conv3d_1_weights = tf.compat.v1.get_variable("weight_conv3d_1", [3, 3, 3, 16, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv3d_1 = tf.nn.conv3d(top_relu, conv3d_1_weights, strides=[1, 2, 2, 2, 1], padding="SAME")
        add = tf.raw_ops.AddV2(x=conv3d_1, y=tf.constant(np.random.randn(32), dtype=tf.float32), name='addv2')
        relu = tf.nn.relu(add)
        conv3d_2_weights = tf.compat.v1.get_variable("weight_conv3d_2", [3, 3, 3, 32, 1],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv3d_2 = tf.nn.conv3d(relu, conv3d_2_weights, strides=[1, 2, 2, 2, 1], padding="SAME")

        out_name = conv3d_2.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 128, 64, 64, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            found_conv_sumadd_fusion = False
            found_conv_biasadd_fusion = False
            for i in output_graph.graph_def.node:
                if i.op == '_QuantizedConv3D':
                    if b'Sum' in i.attr['fused_ops'].list.s:
                        found_conv_sumadd_fusion = True
                    if i.attr['fused_ops'].list.s == [b'BiasAdd', b'Relu', b'Requantize']:
                        found_conv_biasadd_fusion = True
            self.assertEqual(found_conv_sumadd_fusion, False)
            self.assertEqual(found_conv_biasadd_fusion, True)


if __name__ == '__main__':
    unittest.main()
