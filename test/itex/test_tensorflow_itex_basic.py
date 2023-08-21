#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import shutil
import yaml
import platform
from tensorflow.python.platform import gfile
from neural_compressor.adaptor.tf_utils.util import disable_random
from neural_compressor.experimental import Quantization, Benchmark, common
from neural_compressor.adaptor.tf_utils.util import version1_lt_version2, version1_gte_version2

import tensorflow as tf
from tensorflow.python.framework import graph_util

def build_fake_yaml(fake_yaml, save_path, **kwargs):
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open(file=save_path, mode=kwargs['mode'], encoding=kwargs['encoding']) as f:
        yaml.dump(y, f)

@unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0'), "Only supports tf greater 2.7.0")
class TestItexEnabling(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.system("rm *.log")
        fake_yaml_1 = '''
        model:
          name: fake_model_cpu
          framework: tensorflow_itex
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
              path: workspace_1
        '''

        fake_yaml_2 = '''
            model:
              name: fake_model_gpu
              framework: tensorflow_itex
              inputs: input
            device: gpu
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
              performance:
                warmup: 10
                iteration: 100
                configs:
                  cores_per_instance: 1
                  num_of_instance: 1
            tuning:
                strategy:
                  name: basic
                accuracy_criterion:
                  relative: 0.1
                exit_policy:
                  performance_only: True
                workspace:
                  path: workspace_2
            '''

        fake_yaml_3 = '''
            model:
                name: fake_model_default_device
                framework: tensorflow_itex
                inputs: input
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
                    path: workspace_3
        '''
        build_fake_yaml(fake_yaml_1, 'fake_yaml_1.yaml', mode="w", encoding="utf-8")
        build_fake_yaml(fake_yaml_2, 'fake_yaml_2.yaml', mode="w", encoding="utf-8")
        build_fake_yaml(fake_yaml_3, 'fake_yaml_3.yaml', mode="w", encoding="utf-8")

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml_1.yaml')
        os.remove('fake_yaml_2.yaml')
        os.remove('fake_yaml_3.yaml')
        if version1_gte_version2(tf.version.VERSION, '2.8.0'):
            shutil.rmtree('workspace_1')
            shutil.rmtree('workspace_2')
            shutil.rmtree('workspace_3')

    @disable_random()
    def test_itex_convert_basic_default_device(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        add = tf.raw_ops.Add(x=normed, y=normed2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')
        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            quantizer = Quantization('fake_yaml_3.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            dequant_count = 0
            quantize_count = 0
            for i in output_graph.graph_def.node:
                if 'min' in i.name or 'max' in i.name:
                    self.assertEqual(i.op, 'HostConst')
                if i.op == 'HostConst':
                    self.assertTrue('min' in i.name or 'max' in i.name)
                if i.op == 'Dequantize':
                    dequant_count += 1
                if i.op == 'QuantizeV2':
                    quantize_count += 1

            self.assertEqual(dequant_count, 5)
            self.assertEqual(quantize_count, 4)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0'), "Only supports tf greater 2.7.0")
    def test_itex_convert_basic_cpu(self):
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
        add = tf.raw_ops.Add(x=normed, y=normed2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            quantizer = Quantization('fake_yaml_1.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            dequant_count = 0
            quantize_count = 0
            for i in output_graph.graph_def.node:
                if i.op == 'Dequantize':
                    dequant_count += 1
                if i.op == 'QuantizeV2':
                    quantize_count += 1

            self.assertEqual(dequant_count, 5)
            self.assertEqual(quantize_count, 4)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0'), "Only supports tf greater 2.7.0")
    def test_itex_convert_basic_gpu(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        add = tf.raw_ops.Add(x=normed, y=normed2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')
        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            quantizer = Quantization('fake_yaml_2.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            dequant_count = 0
            quantize_count = 0
            for i in output_graph.graph_def.node:
                if i.op == 'HostConst':
                    self.assertTrue('min' in i.name or 'max' in i.name)
                if i.op == 'Dequantize':
                    dequant_count += 1
                if i.op == 'QuantizeV2':
                    quantize_count += 1

            self.assertEqual(dequant_count, 5)
            self.assertEqual(quantize_count, 4)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0'), "Only supports tf greater 2.7.0")
    def test_depthwiseconv2d_case(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.depthwise_conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="VALID")
        out_name = conv.name.split(':')[0]

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml_1.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()
            reshape_counter = 0

            for i in output_graph.graph_def.node:
                if i.op == 'Reshape':
                    reshape_counter += 1
            self.assertEqual(reshape_counter, 2)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0') or \
                     platform.system().lower() == "windows", "Only supports tf greater 2.7.0 and Linux")
    def test_itex_benchmark_gpu(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        add = tf.raw_ops.Add(x=normed, y=normed2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')
        out_name = relu6.name.split(':')[0]
        num_of_instance = 1
        cores_per_instance = 1
        log_file = ''
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            quantizer = Quantization('fake_yaml_2.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            evaluator = Benchmark('fake_yaml_2.yaml')
            evaluator.b_dataloader = common.DataLoader(dataset)
            num_of_instance = evaluator.conf.usr_cfg.evaluation.performance.configs.num_of_instance
            cores_per_instance = evaluator.conf.usr_cfg.evaluation.performance.configs.cores_per_instance
            log_file = '{}_{}_{}.log'.format(num_of_instance, cores_per_instance, 0)
            if gfile.Exists(log_file):
                os.remove(log_file)
            evaluator.model = output_graph
            evaluator('performance')

        found_multi_instance_log = False
        for file_name in os.listdir(os.getcwd()):
            if file_name == log_file:
                found_multi_instance_log = True
                break

        self.assertEqual(found_multi_instance_log, False)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0'), "Only supports tf greater 2.7.0")
    def test_itex_convert_shared_y_pattern_normal_case(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 1, 1, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        #relu2 = tf.nn.relu(normed2)
        add = tf.raw_ops.Add(x=top_relu, y=normed2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            quantizer = Quantization('fake_yaml_1.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            dequant_count = 0
            quantize_count = 0
            for i in output_graph.graph_def.node:
                if i.op == 'Dequantize':
                    dequant_count += 1
                if i.op == 'QuantizeV2':
                    quantize_count += 1

            self.assertEqual(dequant_count, 2)
            self.assertEqual(quantize_count, 2)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0'), "Only supports tf greater 2.7.0")
    def test_itex_convert_share_y_pattern_abnormal_case1(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        relu2 = tf.nn.relu(normed2)
        add = tf.raw_ops.Add(x=relu, y=relu2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            quantizer = Quantization('fake_yaml_1.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            dequant_count = 0
            quantize_count = 0
            for i in output_graph.graph_def.node:
                if i.op == 'Dequantize':
                    dequant_count += 1
                if i.op == 'QuantizeV2':
                    quantize_count += 1

            self.assertEqual(dequant_count, 4)
            self.assertEqual(quantize_count, 3)

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0'), "Only supports tf greater 2.7.0")
    def test_itex_convert_share_y_pattern_abnormal_case2(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed)
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            quantizer = Quantization('fake_yaml_1.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            dequant_count = 0
            quantize_count = 0
            for i in output_graph.graph_def.node:
                if i.op == 'Dequantize':
                    dequant_count += 1
                if i.op == 'QuantizeV2':
                    quantize_count += 1

            self.assertEqual(dequant_count, 2)
            self.assertEqual(quantize_count, 2)

if __name__ == '__main__':
    unittest.main()
