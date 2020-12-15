"""Tests for quantization"""
import numpy as np
import unittest
import os
import yaml
import tensorflow as tf
import shutil
tf.compat.v1.disable_eager_execution()

def build_fake_yaml():
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
            tensorboard: true
            strategy:
              name: basic
            accuracy_criterion:
              relative: 0.01
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_model():
    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session() as sess:
            x = tf.placeholder(tf.float64, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)), name='y')
            op = tf.nn.conv2d(input=x, filter=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(1,3,3,1), name='x')
            y = tf.compat.v1.constant(np.random.random((2,2,1,1)), name='y')
            op = tf.nn.conv2d(input=x, filters=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    return graph

class TestTensorboard(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('saved/history.snapshot')
        os.remove('saved/deploy.yaml')
        os.rmdir('saved')
        shutil.rmtree("runs/")

    def test_run_basic_one_trial(self):
        from lpot import Quantization

        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', (100, 3, 3, 1), label=True)
        dataloader = quantizer.dataloader(dataset)
        quantizer(
            self.constant_graph,
            q_dataloader=dataloader,
            eval_dataloader=dataloader
        )
        self.assertTrue(True if len(os.listdir("./runs/eval")) == 2 else False)

if __name__ == "__main__":
    unittest.main()
