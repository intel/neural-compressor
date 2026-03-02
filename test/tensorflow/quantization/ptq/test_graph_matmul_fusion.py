#
#  -*- coding: utf-8 -*-
#
import os
import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import yaml
from tensorflow.python.framework import dtypes

import neural_compressor
from neural_compressor.tensorflow.algorithms.static_quant.tensorflow import TensorflowQuery
from neural_compressor.tensorflow.utils import disable_random


class TestGraphMatMulFusion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.op_wise_sequences = TensorflowQuery(
            local_config_file=neural_compressor.__path__[0] + "/tensorflow/algorithms/static_quant/tensorflow.yaml"
        ).get_eightbit_patterns()

    @disable_random()
    def test_matmul_biasadd_relu_requantize_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.nn.relu(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                float_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = StaticQuantConfig()
                qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "QuantizedMatMulWithBiasAndReluAndRequantize":
                        found_quantized_matmul = True
                        break
                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_first_matmul_biasadd_relu_fusion(self):
        x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
        y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
        y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
        z = tf.matmul(x, y)
        z = tf.nn.bias_add(z, [1, 2])
        z = tf.nn.relu(z, name="op_to_store")

        with tf.Session() as sess:
            sess.run(z, feed_dict={x: x_data, y: y_data})
            float_graph_def = sess.graph.as_graph_def()

            from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(2, 2), label=True)
            calib_dataloader = BaseDataLoader(dataset, batch_size=2)
            quant_config = StaticQuantConfig()
            qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

            found_quantized_matmul = False
            for i in qmodel.graph_def.node:
                if (
                    i.op == "QuantizeV2"
                    and i.name == "MatMul_eightbit_quantize_x"
                    and i.attr["T"].type == dtypes.quint8
                ):
                    found_quantized_matmul = True
                    break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_requantize_dequantize_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.identity(z, name="op_to_store")
            found_quantized_matmul = False
            if tf.version.VERSION < "2.2.0":
                found_quantized_matmul = True
            else:
                with tf.Session() as sess:
                    sess.run(z, feed_dict={x: x_data, y: y_data})
                    float_graph_def = sess.graph.as_graph_def()

                    from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                    from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                    dataset = DummyDataset(shape=(2, 2), label=True)
                    calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                    quant_config = StaticQuantConfig()
                    qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                    for i in qmodel.graph_def.node:
                        if i.op == "QuantizedMatMulWithBiasAndDequantize":
                            found_quantized_matmul = True
                            break
            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_requantize_dequantize_last_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2], name="op_to_store")
            found_quantized_matmul = False
            if tf.version.VERSION < "2.2.0":
                found_quantized_matmul = True
            else:
                with tf.Session() as sess:
                    sess.run(z, feed_dict={x: x_data, y: y_data})
                    float_graph_def = sess.graph.as_graph_def()

                    from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                    from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                    dataset = DummyDataset(shape=(2, 2), label=True)
                    calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                    quant_config = StaticQuantConfig()
                    qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                    for i in qmodel.graph_def.node:
                        if i.op == "QuantizedMatMulWithBiasAndDequantize" and i.name == "op_to_store":
                            found_quantized_matmul = True
                            break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_disable_matmul_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y, name="no_quant_matmul")
            z = tf.nn.relu6(z, name="op_to_store")
            found_quantized_matmul = False

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                float_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = StaticQuantConfig()
                qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "QuantizedMatMulWithBiasAndDequantize" and i.name == "op_to_store":
                        found_quantized_matmul = True
                        break

            self.assertEqual(found_quantized_matmul, False)

    @disable_random()
    def test_disable_matmul_fusion_with_transpose_b_true(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y, name="no_quant_matmul", transpose_b=True)
            z = tf.nn.relu6(z, name="op_to_store")
            found_quantized_matmul = False

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                float_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = StaticQuantConfig()
                qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "QuantizedMatMulWithBiasAndDequantize" and i.name == "op_to_store":
                        found_quantized_matmul = True
                        break
            self.assertEqual(found_quantized_matmul, False)

    @disable_random()
    @unittest.skipIf(float(tf.__version__[:3]) > 2.7, "only tf lower than 2.8 enable dummy biasadd")
    def test_matmul_with_dummy_biasadd(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y, name="no_quant_matmul")
            z = tf.identity(z, name="op_to_store")
            found_quantized_matmul = True

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                float_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = StaticQuantConfig()
                qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "MatMul":
                        found_quantized_matmul = False
                        break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    @unittest.skipIf(float(tf.__version__[:3]) > 2.7, "only tf lower than 2.8 enable dummy biasadd")
    def test_matmul_with_nan(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            nan_array = np.empty((2, 2), dtype=np.float32)
            nan_array[:] = np.NaN
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            z = tf.matmul(x, nan_array, name="no_quant_matmul")
            z = tf.identity(z, name="op_to_store")
            found_quantized_matmul = True

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data})
                float_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = StaticQuantConfig()
                qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "MatMul":
                        found_quantized_matmul = False
                        break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_with_reshape_transpose(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            transpose = tf.transpose(y, perm=[1, 0])
            reshape = tf.reshape(transpose, [2, 2])
            z = tf.matmul(x, reshape, name="no_quant_matmul")
            z = tf.nn.bias_add(z, [1, 2], name="op_to_store")
            found_quantized_matmul = True

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                float_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = StaticQuantConfig()
                qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "MatMul":
                        found_quantized_matmul = False
                        break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_with_add(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            transpose = tf.transpose(y, perm=[1, 0])
            reshape = tf.reshape(transpose, [2, 2])
            z = tf.matmul(x, reshape, name="no_quant_matmul")
            z = tf.math.add(z, [1, 2], name="op_to_store")
            found_quantized_matmul = True

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                float_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = StaticQuantConfig()
                qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "MatMul":
                        found_quantized_matmul = False
                        break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_requantize_dequantize_fusion_with_softmax(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            biasadd = tf.nn.bias_add(z, [1, 2])
            biasadd1 = tf.nn.bias_add(biasadd, [1, 1])

            y1 = tf.constant(x_data, dtype=tf.float32, shape=[2, 2])
            matmul1 = tf.matmul(biasadd1, y1)

            biasadd2 = tf.nn.bias_add(matmul1, [1, 1])

            z = tf.nn.softmax(biasadd2, name="op_to_store")
            found_quantized_matmul = False
            if tf.version.VERSION < "2.2.0":
                found_quantized_matmul = False
            else:
                with tf.Session() as sess:
                    sess.run(z, feed_dict={x: x_data, y: y_data})
                    float_graph_def = sess.graph.as_graph_def()

                    from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                    from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                    dataset = DummyDataset(shape=(2, 2), label=True)
                    calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                    quant_config = StaticQuantConfig()
                    qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                    count = 0
                    for i in qmodel.model.as_graph_def().node:
                        if i.op == "QuantizedMatMulWithBiasAndDequantize":
                            count += 1
                    found_quantized_matmul = bool(count > 1)

            # TF2.6 has enabled matmul_biasadd_requantize_dequantize_fusion_with_softmax
            if tf.__version__ < "2.6.0":
                self.assertEqual(found_quantized_matmul, False)
            else:
                self.assertEqual(found_quantized_matmul, True)

    def test_matmul_biasadd_relu_non_const_weight(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.matmul(x, x, name="no_quant_matmul")
            biasadd = tf.nn.bias_add(y, [1, 2])
            z = tf.nn.relu(biasadd)
            found_quantized_matmul = True

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data})
                float_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = StaticQuantConfig()
                qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "MatMul":
                        found_quantized_matmul = False
                        break

            self.assertEqual(found_quantized_matmul, False)

    def test_matmul_biasadd_non_const_weight(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.matmul(x, x, name="no_quant_matmul")
            z = tf.nn.bias_add(y, [1, 2])
            found_quantized_matmul = True

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data})
                float_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = StaticQuantConfig()
                qmodel = quantize_model(float_graph_def, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "MatMul":
                        found_quantized_matmul = False
                        break

            self.assertEqual(found_quantized_matmul, False)


if __name__ == "__main__":
    unittest.main()
