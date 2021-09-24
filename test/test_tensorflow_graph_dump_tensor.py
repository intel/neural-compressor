#
#  -*- coding: utf-8 -*-
#
import os
import unittest
import yaml
import tensorflow as tf
import numpy as np

np.random.seed(0)


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
              relative: -0.01
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml_kl():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          optimization:
            arithmetic: False                                 # optional. grappler arithmetic optimizer,default value is True.
          model_wise:
            activation:
                algorithm: kl
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
              relative: 0.99
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_kl.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_model():
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with tf.compat.v1.Session() as sess:
        tf.compat.v1.set_random_seed(0)
        x = tf.compat.v1.placeholder(tf.float32, [1, 30, 30, 1], name="input")
        conv_weights = tf.compat.v1.get_variable("weight", [2, 2, 1, 1],
                                                 initializer=tf.compat.v1.random_normal_initializer())

        beta = tf.compat.v1.get_variable(name='beta',
                                         shape=[1],
                                         initializer=tf.compat.v1.random_normal_initializer())
        gamma = tf.compat.v1.get_variable(name='gamma',
                                          shape=[1],
                                          initializer=tf.compat.v1.random_normal_initializer())

        x = tf.nn.relu(x)
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        conv_bias = tf.compat.v1.layers.batch_normalization(conv1)
        x = tf.nn.relu(conv_bias)
        pool = tf.nn.max_pool(x, ksize=1, strides=[1, 2, 2, 1], padding="SAME")

        final_node = tf.nn.relu(pool, name='op_to_store')
        sess.run(tf.compat.v1.global_variables_initializer())
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=[final_node.name.split(':')[0]])

    graph_def.ParseFromString(constant_graph.SerializeToString())

    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph


class TestGraphDumpToDisk(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        build_fake_yaml()
        build_fake_yaml_kl()
        self.kl_log_path = os.path.join(os.getcwd(), 'saved/kl.log')
        self.calibration_log_path = os.path.join(os.getcwd(), 'saved/requant_min_max.log')

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('fake_yaml_kl.yaml')
        os.remove(self.calibration_log_path)

    def test_dump_tensor_to_disk(self):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 30, 30, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer()

        with open(self.calibration_log_path) as f:
            data = f.readlines()

        found_kl = False
        for i in data:
            if i.find('Relu_1__print__;__KL:') != -1:
                found_kl = True

        self.assertEqual(os.path.exists(self.calibration_log_path), True)
        self.assertGreater(len(data), 1)
        self.assertEqual(found_kl, True)


if __name__ == '__main__':
    unittest.main()
