"""Tests for quantization"""
import numpy as np
import unittest
import shutil
import os
import yaml
from neural_compressor.utils import logger

def build_fake_yaml2():
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
        use_bf16: True
        tuning:
          strategy:
            name: conservative
          accuracy_criterion:
            relative: -0.01
          workspace:
            path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml2.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_model():
    import tensorflow as tf
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

class TestQuantization(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        build_fake_yaml2()
        self.test1_index = -1
        self.test2_index = -1
        self.test3_index = -1
        self.test4_index = -1

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml2.yaml')
        shutil.rmtree('saved', ignore_errors=True)


    def test_conservative_strategy1(self):
        import time
        from neural_compressor.experimental import Quantization, common
        # accuracy increase and performance decrease 
        logger.info("*** Test: accuracy increase and performance decrease.")
        acc_lst =  [1.0, 2.0, 2.1, 3.0, 4.0]
        perf_lst = [2.0, 1.5, 1.0, 0.5, 0.1]
        def _eval(fake_model):
            self.test1_index += 1
            perf = perf_lst[self.test1_index]
            time.sleep(perf)
            return acc_lst[self.test1_index]
            
        quantizer = Quantization('fake_yaml2.yaml')
        quantizer.eval_func = _eval
        dataset = quantizer.dataset('dummy', (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer.fit()
        
    def test_conservative_strategy2(self):
        import time
        from neural_compressor.experimental import Quantization, common
        # accuracy meets and performance disturbance 
        logger.info("*** Test: accuracy meets and performance disturbance.")
        acc_lst =  [5.0, 6.0, 6.0, 6.0, 6.0, 6.0]
        perf_lst = [2.0, 1.5, 1.0, 0.5, 0.1, 0.5]
        def _eval(fake_model):
            self.test2_index += 1
            perf = perf_lst[self.test2_index]
            time.sleep(perf)
            return acc_lst[self.test2_index]
            
        quantizer = Quantization('fake_yaml2.yaml')
        quantizer.eval_func = _eval
        dataset = quantizer.dataset('dummy', (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer.fit()

    def test_conservative_strategy3(self):
        import time
        from neural_compressor.experimental import Quantization, common
        # accuracy disturbance and performance disturbance.
        logger.info("*** Test: accuracy disturbance and performance disturbance.")
        acc_lst =  [5.0, 6.0, 6.0, 4.0, 6.0, 6.0]
        perf_lst = [2.0, 1.5, 1.0, 0.1, 0.5, 1]
        def _eval(fake_model):
            self.test3_index += 1
            perf = perf_lst[self.test3_index]
            time.sleep(perf)
            return acc_lst[self.test3_index]
            
        quantizer = Quantization('fake_yaml2.yaml')
        quantizer.eval_func = _eval
        dataset = quantizer.dataset('dummy', (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer.fit()

    def test_conservative_strategy4(self):
        import time
        from neural_compressor.experimental import Quantization, common
        # accuracy not meet 
        logger.info("*** Test: accuracy not meet.")
        acc_lst =  [5.0, 4.0, 4.0, 4.0]
        perf_lst = [2.0, 1.5, 1.0, 0.1]
        def _eval(fake_model):
            self.test4_index += 1
            perf = perf_lst[self.test4_index]
            time.sleep(perf)
            return acc_lst[self.test4_index]
            
        quantizer = Quantization('fake_yaml2.yaml')
        quantizer.eval_func = _eval
        dataset = quantizer.dataset('dummy', (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer.fit()

if __name__ == "__main__":
    unittest.main()
