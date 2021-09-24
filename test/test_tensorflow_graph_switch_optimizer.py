
import imp
import unittest
import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tf_utils.util import disable_random
from tensorflow.python.ops import control_flow_ops


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


class TestSwitchOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @disable_random()
    def test_switch_optimizer(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        y = tf.compat.v1.placeholder_with_default(True, [], name="place_true")
        
        conv_weights = tf.constant(np.random.random((3,3,16,16)).astype(np.float32), name='y')
        switch_false, switch_true = control_flow_ops.switch(conv_weights, y)
        conv = tf.nn.conv2d(x_pad, switch_true, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed, name='op_to_store')
        out_name = relu.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

        from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.switch_optimizer import SwitchOptimizer

        convert_graph = SwitchOptimizer(output_graph_def).do_transformation()
        found_switch = False
        for node in convert_graph.node:
          if node.op == 'Switch':
            found_switch = True
        self.assertEqual(found_switch, False)

    @disable_random()
    def test_switch_optimizer_invalid(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        y = tf.compat.v1.placeholder_with_default(True, [], name="place_true")
        
        conv_weights = tf.constant(np.random.random((3,3,16,16)).astype(np.float32), name='y')
        switch_false, switch_true = control_flow_ops.switch(conv_weights, y)
        conv = tf.nn.conv2d(x_pad, switch_false, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed, name='op_to_store')
        out_name = relu.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

        from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.switch_optimizer import SwitchOptimizer

        convert_graph = SwitchOptimizer(output_graph_def).do_transformation()
        found_switch = False
        for node in convert_graph.node:
          if node.op == 'Switch':
            found_switch = True
        self.assertEqual(found_switch, True)

if __name__ == "__main__":
    unittest.main()
