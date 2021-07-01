"""Tests for quantization"""
import numpy as np
import unittest
import shutil
import os
import yaml
import tensorflow as tf
if os.getenv('SIGOPT_API_TOKEN') is None or os.getenv('SIGOPT_PROJECT_ID') is None:
    CONDITION = True
else:
    CONDITION = False

def build_fake_yaml(sigopt_api_token,sigopt_project_id):
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: x
          outputs: op2_to_store
        device: cpu
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: sigopt
              sigopt_api_token: {}
              sigopt_project_id: {}
              sigopt_experiment_name: lpot-tune
            accuracy_criterion:
              relative: 0.01
            workspace:
              path: saved
        '''.format(sigopt_api_token, sigopt_project_id)
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_yaml2(sigopt_api_token,sigopt_project_id):
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: x
          outputs: op2_to_store
        device: cpu
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
          strategy:
            name: sigopt
            sigopt_api_token: {}
            sigopt_project_id: {}
            sigopt_experiment_name: lpot-tune
          exit_policy:
            max_trials: 3
          accuracy_criterion:
            relative: -0.01
          workspace:
            path: saved
        '''.format(sigopt_api_token, sigopt_project_id)
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml2.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_model():
    try:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)).astype(np.float32), name='y')
            z = tf.constant(np.random.random((1,1,1,1)).astype(np.float32), name='z')
            op = tf.nn.conv2d(input=tf.nn.relu(x), filters=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')
            op2 = tf.nn.conv2d(input=tf.nn.relu(op), filters=z, strides=[1,1,1,1], padding='VALID', name='op2_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op2_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)).astype(np.float32), name='y')
            z = tf.constant(np.random.random((1,1,1,1)).astype(np.float32), name='z')
            op = tf.nn.conv2d(input=x, filters=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')
            op2 = tf.nn.conv2d(input=op, filters=z, strides=[1,1,1,1], padding='VALID', name='op2_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op2_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    return graph

@unittest.skipIf(CONDITION , "missing the env variables 'SIGOPT_API_TOKEN' or 'SIGOPT_PROJECT_ID'")
class TestQuantization(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sigopt_api_token = os.getenv('SIGOPT_API_TOKEN')
        sigopt_project_id = os.getenv('SIGOPT_PROJECT_ID')
        self.constant_graph = build_fake_model()
        build_fake_yaml(sigopt_api_token,sigopt_project_id)
        build_fake_yaml2(sigopt_api_token,sigopt_project_id)

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('fake_yaml2.yaml')
        shutil.rmtree('saved', ignore_errors=True)

    def test_run_basic_one_trial(self):
        from lpot.experimental import Quantization, common

        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer()


    def test_run_basic_max_trials(self):
        from lpot.experimental import Quantization, common

        quantizer = Quantization('fake_yaml2.yaml')
        dataset = quantizer.dataset('dummy', (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer()

if __name__ == "__main__":
    unittest.main()
