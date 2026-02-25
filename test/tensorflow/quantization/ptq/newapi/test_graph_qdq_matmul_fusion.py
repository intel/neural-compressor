#
#  -*- coding: utf-8 -*-
#
import os
import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import yaml
from tensorflow.python.framework import dtypes

from neural_compressor.tensorflow.utils import disable_random


class TestGraphMatMulFusion(unittest.TestCase):
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
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"Relu",
                        b"Dequantize",
                    ]:
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
            fp32_graph_def = sess.graph.as_graph_def()

            from neural_compressor.tensorflow import Model, quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(2, 2), label=True)
            calib_dataloader = BaseDataLoader(dataset, batch_size=2)
            quant_config = {
                "static_quant": {
                    "global": {
                        "weight_dtype": "int8",
                        "weight_sym": True,
                        "weight_granularity": "per_tensor",
                        "weight_algorithm": "minmax",
                    },
                }
            }
            fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
            qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

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

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [b"BiasAdd", b"Dequantize"]:
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
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if (
                        i.op == "_QuantizedMatMul"
                        and i.name == "op_to_store"
                        and i.attr["fused_ops"].list.s == [b"BiasAdd", b"Dequantize"]
                    ):
                        found_quantized_matmul = True
                        break
            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_fusion_with_transpose_b_true(self):
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
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul":
                        found_quantized_matmul = True
                        break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_dummybiasadd_relu6_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y, name="quant_matmul")
            z = tf.nn.relu6(z, name="op_to_store")
            found_quantized_matmul = False

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.name == "op_to_store":
                        found_quantized_matmul = True
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
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

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
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

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
                    fp32_graph_def = sess.graph.as_graph_def()

                    from neural_compressor.tensorflow import quantize_model
                    from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                    dataset = DummyDataset(shape=(2, 2), label=True)
                    calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                    quant_config = {
                        "static_quant": {
                            "global": {
                                "weight_dtype": "int8",
                                "weight_sym": True,
                                "weight_granularity": "per_tensor",
                                "weight_algorithm": "minmax",
                            },
                        }
                    }
                    qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

                    count = 0
                    for i in qmodel.model.as_graph_def().node:
                        if i.op == "_QuantizedMatMul":
                            count += 1
                    found_quantized_matmul = bool(count > 1)

            self.assertEqual(found_quantized_matmul, False)

    def test_matmul_biasadd_relu_non_const_weight(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.matmul(x, x, name="quant_matmul_non_const_weight")
            biasadd = tf.nn.bias_add(y, [1, 2])
            z = tf.nn.relu(biasadd)
            found_quantized_matmul = True

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "MatMul":
                        found_quantized_matmul = False
                        break

            self.assertEqual(found_quantized_matmul, True)

    def test_matmul_biasadd_non_const_weight(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.matmul(x, x, name="quant_matmul_non_const_weight")
            z = tf.nn.bias_add(y, [1, 2])
            found_quantized_matmul = True

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "MatMul":
                        found_quantized_matmul = False
                        break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
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
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "MatMul":
                        found_quantized_matmul = False
                        break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_first_matmul_addv2_relu_fusion(self):
        x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
        y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
        y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
        a = tf.matmul(x, y)
        b = tf.matmul(x, y)
        c = tf.nn.relu(b)
        add = tf.raw_ops.AddV2(x=a, y=c, name="addv2")
        z = tf.nn.relu(add, name="op_to_store")

        with tf.Session() as sess:
            sess.run(z, feed_dict={x: x_data, y: y_data})
            fp32_graph_def = sess.graph.as_graph_def()

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(2, 2), label=True)
            calib_dataloader = BaseDataLoader(dataset, batch_size=2)
            quant_config = {
                "static_quant": {
                    "global": {
                        "weight_dtype": "int8",
                        "weight_sym": True,
                        "weight_granularity": "per_tensor",
                        "weight_algorithm": "minmax",
                    },
                }
            }
            qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

            found_quantized_matmul = False
            for i in qmodel.graph_def.node:
                if i.op == "_QuantizedMatMul":
                    found_quantized_matmul = True
                    break

            self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_relu6_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.nn.relu6(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"Relu6",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_leakyrelu_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.nn.leaky_relu(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"LeakyRelu",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_geluapproximate_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.nn.gelu(z, approximate=True, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"GeluApproximate",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_geluexact_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.nn.gelu(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"GeluExact",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_elu_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.nn.elu(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [b"BiasAdd", b"Elu", b"Dequantize"]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_tanh_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.math.tanh(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"Tanh",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_sigmoid_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.math.sigmoid(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"Sigmoid",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_dummy_biasadd_relu_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y, name="quant_matmul")
            z = tf.nn.relu(z, name="op_to_store")
            found_quantized_matmul = False

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"Relu",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_dummy_biasadd_relu6_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.relu6(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"Relu6",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_dummy_biasadd_leakyrelu_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.leaky_relu(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"LeakyRelu",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_dummy_biasadd_geluapproximate_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.gelu(z, approximate=True, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"GeluApproximate",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_dummy_biasadd_geluexact_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.gelu(z, approximate=False, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"GeluExact",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_dummy_biasadd_elu_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.elu(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [b"BiasAdd", b"Elu", b"Dequantize"]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_dummy_biasadd_tanh_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.math.tanh(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"Tanh",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_dummy_biasadd_sigmoid_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.math.sigmoid(z, name="op_to_store")
            found_quantized_matmul = False
            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [
                        b"BiasAdd",
                        b"Sigmoid",
                        b"Dequantize",
                    ]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_add_const_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            transpose = tf.transpose(y, perm=[1, 0])
            reshape = tf.reshape(transpose, [2, 2])
            z = tf.matmul(x, reshape, name="quant_matmul")
            z = tf.math.add(z, [1, 2], name="op_to_store")
            found_quantized_matmul = False

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [b"BiasAdd", b"Dequantize"]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_add_non_const_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            transpose = tf.transpose(y, perm=[1, 0])
            reshape = tf.reshape(transpose, [2, 2])
            z = tf.matmul(x, reshape, name="quant_matmul")
            z = tf.math.add(z, x, name="addv2")
            z = tf.nn.relu(z, name="op_to_store")
            found_quantized_matmul = False

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [b"Dequantize"]:
                        found_quantized_matmul = True
                        break
                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_add_const_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.math.add(z, [1, 2], name="op_to_store")
            found_quantized_matmul = False

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [b"BiasAdd", b"Dequantize"]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)

    @disable_random()
    def test_matmul_biasadd_add_non_const_fusion(self):
        g = tf.Graph()
        with g.as_default():
            x_data = np.array([[0.1, 0.2], [0.2, 0.3]])
            y_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x = tf.placeholder(tf.float32, shape=[2, 2], name="x")
            y = tf.constant(y_data, dtype=tf.float32, shape=[2, 2])
            z = tf.matmul(x, y)
            z = tf.nn.bias_add(z, [1, 2])
            z = tf.math.add(z, x, name="op_to_store")
            found_quantized_matmul = False

            with tf.Session() as sess:
                sess.run(z, feed_dict={x: x_data, y: y_data})
                fp32_graph_def = sess.graph.as_graph_def()

                from neural_compressor.tensorflow import Model, quantize_model
                from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

                dataset = DummyDataset(shape=(2, 2), label=True)
                calib_dataloader = BaseDataLoader(dataset, batch_size=2)
                quant_config = {
                    "static_quant": {
                        "global": {
                            "weight_dtype": "int8",
                            "weight_sym": True,
                            "weight_granularity": "per_tensor",
                            "weight_algorithm": "minmax",
                        },
                    }
                }
                fp32_model = Model(fp32_graph_def, conf={"performance_only": True})
                qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)

                for i in qmodel.graph_def.node:
                    if i.op == "_QuantizedMatMul" and i.attr["fused_ops"].list.s == [b"BiasAdd", b"Dequantize"]:
                        found_quantized_matmul = True
                        break

                self.assertEqual(found_quantized_matmul, True)


if __name__ == "__main__":
    unittest.main()
