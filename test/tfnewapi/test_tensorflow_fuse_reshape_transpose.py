
import imp
import unittest
import os
from numpy.core.fromnumeric import squeeze
import yaml
import numpy as np
from neural_compressor.adaptor.tf_utils.util import disable_random

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import graph_util


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
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


class TestFuseReshapeTransposeOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @disable_random()
    def test_fuse_enter_reshape_transpose(self):
        x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
        y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=[2, 2], name='x')
        y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
        enter = tf.raw_ops.Enter(data=y, frame_name='test')
        enter_perm = tf.raw_ops.Enter(data=[1, 0], frame_name='test', is_constant=True)
        transpose = tf.transpose(enter, perm=enter_perm)
        enter_reshape = tf.raw_ops.Enter(data=[2, 2], frame_name='test', is_constant=True)
        reshape = tf.reshape(transpose, enter_reshape)
        x_enter = tf.raw_ops.Enter(data=x, frame_name='test')
        z = tf.raw_ops.MatMul(a=x_enter, b=reshape, name='matmul_1')
        z = tf.raw_ops.Exit(data=z)
        found_quantized_matmul = True
        found_transpose = False
        found_reshape = False

        with tf.Session() as sess:
            sess.run(z, feed_dict={x: x_data, y: y_data})
            float_graph_def = sess.graph.as_graph_def()

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(2, 2), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=2)
            quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=2)
            quantizer.model = float_graph_def
            output_graph = quantizer.fit()
            for i in output_graph.graph_def.node:
                if i.op == 'MatMul':
                    found_quantized_matmul = False
                if i.op == 'Transpose':
                    found_transpose = True
                if i.op == 'Reshape':
                    found_reshape = True
            self.assertEqual(found_quantized_matmul, True)
            self.assertEqual(found_transpose, False)
            self.assertEqual(found_reshape, False)

    @disable_random()
    def test_fuse_reshape_transpose(self):
        x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
        y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=[2, 2], name='x')
        y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
        transpose = tf.transpose(y, perm=[1, 0])
        reshape = tf.reshape(transpose, [2, 2])
        z = tf.raw_ops.MatMul(a=x, b=reshape, name='matmul_2')
        z = tf.nn.bias_add(z, [1, 2], name='op_to_store')
        found_quantized_matmul = True
        found_transpose = False
        found_reshape = False

        with tf.Session() as sess:
            sess.run(z, feed_dict={x: x_data, y: y_data})
            float_graph_def = sess.graph.as_graph_def()

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(2, 2), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=2)
            quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=2)
            quantizer.model = float_graph_def
            output_graph = quantizer.fit()

            for i in output_graph.graph_def.node:
                if i.op == 'MatMul':
                    found_quantized_matmul = False
                if i.op == 'Transpose':
                    found_transpose = True
                if i.op == 'Reshape':
                    found_reshape = True
            self.assertEqual(found_quantized_matmul, True)
            self.assertEqual(found_transpose, False)
            self.assertEqual(found_reshape, False)

if __name__ == "__main__":
    unittest.main()
