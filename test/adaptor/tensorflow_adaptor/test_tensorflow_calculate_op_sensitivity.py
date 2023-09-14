import os
import shutil
import unittest

import numpy as np
import tensorflow as tf


def build_msev2_yaml():
    mse_yaml = """
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
    """
    with open("mse_yaml.yaml", "w", encoding="utf-8") as f:
        f.write(mse_yaml)


def build_fake_model():
    try:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1, 3, 3, 1), name="x")
            y = tf.constant(np.random.random((2, 2, 1, 1)).astype(np.float32), name="y")
            z = tf.constant(np.random.random((1, 1, 1, 1)).astype(np.float32), name="z")
            op = tf.nn.conv2d(input=x, filters=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")
            op2 = tf.nn.conv2d(
                input=op,
                filters=z,
                strides=[1, 1, 1, 1],
                padding="VALID",
            )
            last_identity = tf.identity(op2, name="op2_to_store")
            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ["op2_to_store"]
            )

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1, 3, 3, 1), name="x")
            y = tf.constant(np.random.random((2, 2, 1, 1)).astype(np.float32), name="y")
            z = tf.constant(np.random.random((1, 1, 1, 1)).astype(np.float32), name="z")
            op = tf.nn.conv2d(input=x, filters=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")
            op2 = tf.nn.conv2d(input=op, filters=z, strides=[1, 1, 1, 1], padding="VALID")
            last_identity = tf.identity(op2, name="op2_to_store")

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ["op2_to_store"]
            )

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    return graph


class TestGetOutputTensor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_msev2_yaml()
        self.model = build_fake_model()

    @classmethod
    def tearDownClass(self):
        os.remove("mse_yaml.yaml")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_get_output_op_names(self):
        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization("mse_yaml.yaml")
        dataset = quantizer.dataset("dummy", (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.model
        qmodel = quantizer.fit()

        self.assertEqual(quantizer.strategy.adaptor.get_output_op_names(qmodel), ["Conv2D_dummy_biasadd"])

    def test_calculate_op_sensitivity(self):
        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization("mse_yaml.yaml")
        quantizer.model = self.model
        dataset = quantizer.dataset("dummy", (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.pre_process()

        dataloader = quantizer._calib_dataloader
        strategy = quantizer.strategy
        adaptor = strategy.adaptor
        tune_cfg_generator = strategy.next_tune_cfg()
        tune_cfg = strategy._tune_cfg_converter(next(tune_cfg_generator))
        output_op_names = ["Conv2D_dummy_biasadd"]

        op_sensitivity = adaptor.calculate_op_sensitivity(
            model=quantizer.model,
            dataloader=dataloader,
            tune_cfg=tune_cfg,
            output_op_names=output_op_names,
            confidence_batches=1,
            fallback=True,
        )
        self.assertIn(("op_to_store", "conv2d"), op_sensitivity)
        self.assertIn(("Conv2D", "conv2d"), op_sensitivity)

        tune_cfg["op"][("op_to_store", "conv2d")] = {
            "activation": {"dtype": "fp32", "quant_mode": "fp32"},
            "weight": {"dtype": "fp32"},
        }

        op_sensitivity = adaptor.calculate_op_sensitivity(
            model=quantizer.model,
            dataloader=dataloader,
            tune_cfg=tune_cfg,
            output_op_names=output_op_names,
            confidence_batches=1,
            fallback=True,
        )
        self.assertNotIn(("op_to_store", "conv2d"), op_sensitivity)
        self.assertIn(("Conv2D", "conv2d"), op_sensitivity)


if __name__ == "__main__":
    unittest.main()
