"""Tests for lpot quantization"""
import numpy as np
import unittest
import os
import yaml
import tensorflow as tf
import importlib
import shutil
     
def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: x
          outputs: identity
        quantization:
          recipes:
            fast_bias_correction: True
          calibration:
            sampling_size: 10
        device: cpu
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
          accuracy_criterion:
            relative: 0.01        
          workspace:
            path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_yaml2():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: x
          outputs: identity
        quantization:
          recipes:
            weight_correction: True
          calibration:
            sampling_size: 10
        device: cpu
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
          accuracy_criterion:
            relative: 0.01
          workspace:
            path: saved
            resume: ./saved/history.snapshot
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml2.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_model():
    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)), dtype=tf.float32, name='y')
            relu_0 = tf.nn.relu(x, name='relu')
            conv = tf.nn.conv2d(input=relu_0, filters=y, strides=[1,1,1,1], padding='VALID', name='conv')
            bias = tf.Variable(tf.ones([1], tf.float32))
            conv_add = tf.nn.bias_add(conv, bias, name='bias_add')
            relu = tf.nn.relu(conv_add)
            op = tf.identity(relu, name='identity')

            sess.run(tf.global_variables_initializer())
            from tensorflow.compat.v1.graph_util import convert_variables_to_constants
            constant_graph = convert_variables_to_constants(sess, sess.graph_def, ['identity'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1,3,3,1), name='x')
            y = tf.compat.v1.constant(np.random.random((2,2,1,1)), dtype=tf.float32, name='y')
            relu_0 = tf.nn.relu(x, name='relu')
            conv = tf.nn.conv2d(input=relu_0, filters=y, strides=[1,1,1,1], padding='VALID', name='conv')
            bias = tf.Variable(tf.ones([1], tf.float32))
            conv_add = tf.nn.bias_add(conv, bias, name='bias_add')
            relu = tf.nn.relu(conv_add)
            op = tf.identity(relu, name='identity')

            sess.run(tf.compat.v1.global_variables_initializer())
            from tensorflow.compat.v1.graph_util import convert_variables_to_constants
            constant_graph = convert_variables_to_constants(sess, sess.graph_def, ['identity'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    return graph

class TestQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        build_fake_yaml()
        build_fake_yaml2()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('fake_yaml2.yaml')    
        shutil.rmtree('./saved', ignore_errors=True)

    def test_fast_bias_correction(self):
        from lpot.experimental import Quantization, common
        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        output_graph = quantizer()

    def test_weight_correction(self):
        from lpot.experimental import Quantization, common
        quantizer = Quantization('fake_yaml2.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        output_graph = quantizer()


if __name__ == "__main__":
    unittest.main()
