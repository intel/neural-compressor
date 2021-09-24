import os
import unittest
import yaml
import numpy as np
import tensorflow as tf
import shutil
from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tensorflow import TensorFlowAdaptor
from neural_compressor.adaptor.tf_utils.util import disable_random

def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            accuracy_criterion:
              relative: 0.0001
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()
class TestSetTensor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        
    @disable_random()
    def test_fp32bias(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                            initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")

        conv_bias = tf.compat.v1.get_variable("bias", [16], dtype=tf.float32,
                                            initializer=tf.compat.v1.random_normal_initializer())

        conv_bias = tf.math.add(conv, conv_bias)
        relu6 = tf.nn.relu6(conv_bias, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization("./fake_yaml.yaml")
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = constant_graph
            q_model = quantizer()
            
            framework_specific_info = {'device': 'cpu', 'workspace_path': 'saved',\
                    'random_seed': 1978, 'inputs': ['input'], 'outputs': ['op_to_store'], \
                    'approach': 'post_training_static_quant'}
            adaptor = TensorFlowAdaptor(framework_specific_info)
            adaptor.set_tensor(q_model, {'bias': np.random.random(16)})

            from tensorflow.core.framework import attr_value_pb2
            from tensorflow.python.framework import dtypes
            for node in q_model.graph_def.node:
                if node.name == 'bias':
                    self.assertEqual(node.attr['dtype'], attr_value_pb2.AttrValue(
                    type=dtypes.float32.as_datatype_enum))

    @disable_random()
    def test_int32bias(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                            initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")

        conv_bias = tf.compat.v1.get_variable("bias", [16], dtype=tf.float32)

        conv_bias = tf.math.add(conv, conv_bias)
        relu6 = tf.nn.relu6(conv_bias, name='relu_0')

        conv_weights1 = tf.compat.v1.get_variable("weight1", [3, 3, 16, 16],
                                            initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(relu6, conv_weights1, strides=[1, 2, 2, 1], padding="VALID")

        conv_bias1 = tf.compat.v1.get_variable("bias1", [16], dtype=tf.float32)

        conv_bias1 = tf.math.add(conv1, conv_bias1)
        relu6 = tf.nn.relu6(conv_bias1, name='relu_1')

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                            initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(relu6, conv_weights2, strides=[1, 2, 2, 1], padding="VALID")

        conv_bias2 = tf.compat.v1.get_variable("bias2", [16], dtype=tf.float32)

        conv_bias2 = tf.math.add(conv2, conv_bias2)
        relu6 = tf.nn.relu6(conv_bias2, name='op_to_store')
        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

            for i in constant_graph.node:
                if i.op.find('Add') != -1:
                    i.op = 'Add'

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization("./fake_yaml.yaml")
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = constant_graph
            q_model = quantizer()

            framework_specific_info = {'device': 'cpu', 'workspace_path': 'saved',\
                    'random_seed': 1978, 'inputs': ['input'], 'outputs': ['op_to_store'], \
                    'approach': 'post_training_static_quant'}
            adaptor = TensorFlowAdaptor(framework_specific_info)
            adaptor.set_tensor(q_model, {'bias1': np.random.randint(6,size=2, dtype='int32')})
            from tensorflow.core.framework import attr_value_pb2
            from tensorflow.python.framework import dtypes
            for node in q_model.graph_def.node:
                if node.name == 'bias2':
                    self.assertEqual(node.attr['dtype'], attr_value_pb2.AttrValue(
                    type=dtypes.qint32.as_datatype_enum))


if __name__ == "__main__":
    unittest.main()
