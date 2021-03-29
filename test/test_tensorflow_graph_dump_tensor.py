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
        conv_bias = tf.compat.v1.get_variable("bias", [1],
                                              initializer=tf.compat.v1.random_normal_initializer())
        beta = tf.compat.v1.get_variable(name='beta',
                                         shape=[1],
                                         initializer=tf.compat.v1.random_normal_initializer())
        gamma = tf.compat.v1.get_variable(name='gamma',
                                          shape=[1],
                                          initializer=tf.compat.v1.random_normal_initializer())

        x = tf.nn.relu(x)
        pool = tf.nn.max_pool(x, ksize=1, strides=[1, 2, 2, 1], padding="SAME")
        conv1 = tf.nn.conv2d(pool, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name='last')
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        x = tf.nn.relu(conv_bias)
        final_node = tf.nn.relu(x, name='op_to_store')
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
        # os.remove(self.kl_log_path)
        os.remove(self.calibration_log_path)

    # def test_kl(self):
    #     import tensorflow.compat.v1 as tf
    #     tf.disable_v2_behavior()

    #     from lpot.experimental import Quantization

    #     quantizer = Quantization('fake_yaml_kl.yaml')
    #     dataset = quantizer.dataset('dummy', shape=(100, 30, 30, 1), label=True)
    #     dataloader = quantizer.dataloader(dataset)
    #     output_graph = quantizer(
    #         self.constant_graph,
    #         q_dataloader=dataloader,
    #         eval_dataloader=dataloader
    #     )

    #     with open(self.calibration_log_path) as f:
    #         data = f.readlines()

    #     found_min_str = False
    #     found_max_str = False
    #     for i in data:
    #         if i.find('__print__;__max') != -1:
    #             found_max_str = True
    #         if i.find('__print__;__min') != -1:
    #             found_min_str = True

    #     self.assertEqual(os.path.exists(self.kl_log_path), True)

    #     self.assertEqual(os.path.exists(self.calibration_log_path), True)
    #     self.assertGreater(len(data), 1)
    #     self.assertEqual(found_min_str, True)
    #     self.assertEqual(found_max_str, True)

    def test_dump_tensor_to_disk(self):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        from lpot.experimental import Quantization, common

        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 30, 30, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer()

        with open(self.calibration_log_path) as f:
            data = f.readlines()

        found_min_str = False
        found_max_str = False
        for i in data:
            if i.find('__print__;__max') != -1:
                found_max_str = True
            if i.find('__print__;__min') != -1:
                found_min_str = True

        self.assertEqual(os.path.exists(self.calibration_log_path), True)
        self.assertGreater(len(data), 1)
        self.assertEqual(found_min_str, True)
        self.assertEqual(found_max_str, True)


if __name__ == '__main__':
    unittest.main()
