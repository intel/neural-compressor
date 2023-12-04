"""Tests for quantization."""
import os
import shutil
import unittest

import numpy as np


def build_fake_model():
    import tensorflow as tf

    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session() as sess:
            x = tf.placeholder(tf.float64, shape=(1, 3, 3, 1), name="x")
            y = tf.constant(np.random.random((2, 2, 1, 1)), name="y")
            op = tf.nn.conv2d(input=x, filter=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["op_to_store"])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(1, 3, 3, 1), name="x")
            y = tf.compat.v1.constant(np.random.random((2, 2, 1, 1)), name="y")
            op = tf.nn.conv2d(input=x, filters=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ["op_to_store"]
            )

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    return graph


class TestRandomStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("saved", ignore_errors=True)

    def test_ru_random_one_trial(self):
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"]((100, 3, 3, 1), label=True)
        dataloader = DATALOADERS["tensorflow"](dataset)

        # tuning and accuracy criterion
        tune_cri = TuningCriterion(strategy="random", max_trials=1)
        acc_cri = AccuracyCriterion(tolerable_loss=0.01)

        conf = PostTrainingQuantConfig(quant_level=1, tuning_criterion=tune_cri, accuracy_criterion=acc_cri)

        def fake_eval(model):
            return 1

        q_model = fit(model=self.constant_graph, conf=conf, calib_dataloader=dataloader, eval_func=fake_eval)
        self.assertNotEqual(q_model, None)

    def test_ru_random_max_trials(self):
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"]((100, 3, 3, 1), label=True)
        dataloader = DATALOADERS["tensorflow"](dataset)

        # tuning and accuracy criterion
        tune_cri = TuningCriterion(strategy="random", max_trials=3)
        acc_cri = AccuracyCriterion(tolerable_loss=0.01)

        acc = [0, 1, 0.9, 1]

        def fake_eval(model):
            acc.pop(0)
            return acc[0]

        conf = PostTrainingQuantConfig(quant_level=1, tuning_criterion=tune_cri, accuracy_criterion=acc_cri)
        q_model = fit(model=self.constant_graph, conf=conf, calib_dataloader=dataloader, eval_func=fake_eval)
        self.assertNotEqual(q_model, None)


if __name__ == "__main__":
    unittest.main()
