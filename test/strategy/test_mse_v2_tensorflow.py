import os
import shutil
import unittest
import tensorflow as tf
import numpy as np
from neural_compressor.experimental import Quantization, common


def build_mse_yaml():
    mse_yaml = '''
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
            name: mse_v2
        accuracy_criterion:
            relative:  0.01
        exit_policy:
            max_trials: 10
            timeout: 3600
        random_seed: 9527
    '''
    with open('mse_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(mse_yaml)

def build_fake_model():
    try:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)).astype(np.float32), name='y')
            z = tf.constant(np.random.random((1,1,1,1)).astype(np.float32), name='z')
            op = tf.nn.conv2d(input=x, filters=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')
            op2 = tf.nn.conv2d(input=op, filters=z, strides=[1,1,1,1], padding='VALID', )
            last_identity = tf.identity(op2, name='op2_to_store')
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
            op2 = tf.nn.conv2d(input=op, filters=z, strides=[1,1,1,1], padding='VALID')
            last_identity = tf.identity(op2, name='op2_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op2_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    return graph

i = 0
def eval_func(model):
    #               1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    eval_list = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    
    global i
    i += 1
    return eval_list[i]


class TestMSEV2Strategy_Tensorflow(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_mse_yaml()
        self.model = build_fake_model()

    @classmethod
    def tearDownClass(self):
        os.remove('mse_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_quantization_saved(self):
        
        quantizer = Quantization('mse_yaml.yaml')
        
        quantizer.model = self.model
        dataset = quantizer.dataset('dummy', (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.eval_func = eval_func
        q_model = quantizer.fit()
        q_model.save('./saved')

if __name__ == "__main__":
    unittest.main()
