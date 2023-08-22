import unittest
import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from neural_compressor.adaptor.tf_utils.util import disable_random


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


class TestDqCastFusion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        os.environ['FORCE_BF16'] = '1'

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @disable_random()
    def test_dq_all_outputs_bf16(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        conv_weights = tf.constant(np.random.random((1, 3, 16, 16)).astype(np.float32), name='y')
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        conv_reshape1 = tf.reshape(conv, [1,28,27,16])
        conv_reshape2 = tf.reshape(conv, [1,28,27,16])
        out = tf.math.add(conv_reshape1, conv_reshape2, name='op_to_store')
        out_name = out.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset(
            'dummy', shape=(100, 56, 56, 16))
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = output_graph_def
        output_graph = quantizer.fit()
        found_cast = False
        for node in output_graph.graph_def.node:
            if node.op == 'Cast':
                found_cast = True
                break
        self.assertEqual(found_cast, False)
     
if __name__ == "__main__":
    unittest.main()