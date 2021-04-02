#
#  -*- coding: utf-8 -*-
#
import os
import unittest
import yaml
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import dtypes
from lpot.adaptor.tensorflow import TensorflowQuery
from lpot.adaptor.tf_utils.util import disable_random

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
            strategy:
              name: basic
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


class TestGraphMatMulFusion(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        self.op_wise_sequences = TensorflowQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "../lpot/adaptor/tensorflow.yaml")).get_eightbit_patterns()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @disable_random()
    def test_matmul_biasadd_relu_requantize_fusion(self):

        g = tf.Graph()
        with g.as_default():

            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float)
            x = tf.placeholder(tf.float32, shape=[2, 2], name='x')
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.nn.relu(z, name='op_to_store')
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                float_graph_def = sess.graph.as_graph_def()

                from lpot.experimental import Quantization, common
                quantizer = Quantization('fake_yaml.yaml')
                dataset = quantizer.dataset('dummy', shape=(2, 2), label=True)
                quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=2)
                quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=2)
                quantizer.model = float_graph_def
                output_graph = quantizer()

                for i in output_graph.graph_def.node:
                    if i.op == 'QuantizedMatMulWithBiasAndReluAndRequantize':
                        found_quantized_matmul = True
                        break
                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_first_matmul_biasadd_relu_fusion(self):
        x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
        y_data = np.array([[1, 2], [3, 4]], dtype=np.float)
        x = tf.placeholder(tf.float32, shape=[2, 2], name='x')
        y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
        z = tf.matmul(x, y)
        z = tf.nn.bias_add(z, [1, 2])
        z = tf.nn.relu(z,  name='op_to_store')

        with tf.Session() as sess:

            sess.run(z, feed_dict={x: x_data, y: y_data})
            float_graph_def = sess.graph.as_graph_def()

            from lpot.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(2, 2), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=2)
            quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=2)
            quantizer.model = float_graph_def
            output_graph = quantizer()

            found_quantized_matmul = False
            for i in output_graph.graph_def.node:
                if i.op == 'QuantizeV2' and i.name == 'MatMul_eightbit_quantize_x' and i.attr["T"].type == dtypes.quint8:
                    found_quantized_matmul = True
                    break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_requantize_dequantize_fusion(self):
        g = tf.Graph()
        with g.as_default():

            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float)
            x = tf.placeholder(tf.float32, shape=[2, 2], name='x')
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.identity(z, name='op_to_store')
            found_quantized_matmul = False
            if tf.version.VERSION < "2.2.0":
                found_quantized_matmul = True
            else:
                with tf.Session() as sess:
                    sess.run(z, feed_dict={x: x_data, y: y_data})
                    float_graph_def = sess.graph.as_graph_def()

                    from lpot.experimental import Quantization, common
                    quantizer = Quantization('fake_yaml.yaml')
                    dataset = quantizer.dataset('dummy', shape=(2, 2), label=True)
                    quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=2)
                    quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=2)
                    quantizer.model = float_graph_def
                    output_graph = quantizer()

                    for i in output_graph.graph_def.node:
                        if i.op == 'QuantizedMatMulWithBiasAndDequantize':
                            found_quantized_matmul = True
                            break
            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_requantize_dequantize_last_fusion(self):
        g = tf.Graph()
        with g.as_default():

            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float)
            x = tf.placeholder(tf.float32, shape=[2, 2], name='x')
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2], name='op_to_store')
            found_quantized_matmul = False
            if tf.version.VERSION < "2.2.0":
                found_quantized_matmul = True
            else:
                with tf.Session() as sess:
                    sess.run(z, feed_dict={x: x_data, y: y_data})
                    float_graph_def = sess.graph.as_graph_def()

                    from lpot.experimental import Quantization, common
                    quantizer = Quantization('fake_yaml.yaml')
                    dataset = quantizer.dataset('dummy', shape=(2, 2), label=True)
                    quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=2)
                    quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=2)
                    quantizer.model = float_graph_def
                    output_graph = quantizer()

                    for i in output_graph.graph_def.node:
                        if i.op == 'QuantizedMatMulWithBiasAndDequantize' and i.name == 'op_to_store':
                            found_quantized_matmul = True
                            break
            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_disable_matmul_fusion(self):
        g = tf.Graph()
        with g.as_default():

            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float)
            x = tf.placeholder(tf.float32, shape=[2, 2], name='x')
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y, name='no_quant_matmul')
            z = tf.nn.relu6(z, name='op_to_store')
            found_quantized_matmul = False

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                float_graph_def = sess.graph.as_graph_def()

                from lpot.experimental import Quantization, common
                quantizer = Quantization('fake_yaml.yaml')
                dataset = quantizer.dataset('dummy', shape=(2, 2), label=True)
                quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=2)
                quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=2)
                quantizer.model = float_graph_def
                output_graph = quantizer()

                for i in output_graph.graph_def.node:
                    if i.op == 'QuantizedMatMulWithBiasAndDequantize' and i.name == 'op_to_store':
                        found_quantized_matmul = True
                        break
            self.assertEqual(found_quantized_matmul, False)

    @disable_random()
    def test_matmul_biasadd_requantize_dequantize_fusion_with_softmax(self):
        g = tf.Graph()
        with g.as_default():

            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float)
            x = tf.placeholder(tf.float32, shape=[2, 2], name='x')
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            biasadd = tf.nn.bias_add(z, [1, 2])
            biasadd1 = tf.nn.bias_add(biasadd, [1, 1])

            y1 = tf.constant(x_data, dtype=tf.float32, shape=[2, 2])
            matmul1 = tf.matmul(biasadd1, y1)

            biasadd2 = tf.nn.bias_add(matmul1, [1, 1])

            z = tf.nn.softmax(biasadd2, name='op_to_store')
            found_quantized_matmul = False
            if tf.version.VERSION < "2.2.0":
                found_quantized_matmul = False
            else:
                with tf.Session() as sess:
                    sess.run(z, feed_dict={x: x_data, y: y_data})
                    float_graph_def = sess.graph.as_graph_def()

                    from lpot.experimental import Quantization, common
                    quantizer = Quantization('fake_yaml.yaml')
                    dataset = quantizer.dataset('dummy', shape=(2, 2), label=True)
                    quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=2)
                    quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=2)
                    quantizer.model = float_graph_def
                    output_graph = quantizer()

                    count=0
                    for i in output_graph.model.as_graph_def().node:
                        if i.op == 'QuantizedMatMulWithBiasAndDequantize':
                            count += 1
                    found_quantized_matmul = bool(count > 1)
            self.assertEqual(found_quantized_matmul, False)


if __name__ == '__main__':
    unittest.main()
