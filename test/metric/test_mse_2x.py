"""Tests for mse metric/strategy."""

import os
import shutil
import unittest

import numpy as np

from neural_compressor.utils import logger


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
    logger.info("Created model graph.")
    return graph


def build_ox_model():
    import torch
    import torchvision

    path = "mb_v2.onnx"
    model = torchvision.models.mobilenet_v2()

    x = torch.randn(100, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    torch.onnx.export(
        model,
        x,
        path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info("Created onnx model.")


class dataset:
    def __init__(self):
        self.data = []
        self.label = []
        for i in range(10):
            self.data.append(np.zeros((3, 224, 224)).astype(np.float32))
            self.label.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class TestMetric(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_ox_model()
        self.constant_graph = None

    @classmethod
    def tearDownClass(self):
        os.remove("mb_v2.onnx")
        shutil.rmtree("nc_workspace", ignore_errors=True)
        shutil.rmtree("saved", ignore_errors=True)

    def _get_tf_model(self):
        self.constant_graph = self.constant_graph or build_fake_model()
        return self.constant_graph

    def _test_tf_model_helper(self, config, eval_func=None, eval_metric=None, eval_dataloader=None):
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)

        q_model = fit(
            model=self._get_tf_model(),
            conf=config,
            calib_dataloader=dataloader,
            eval_func=eval_func,
            eval_dataloader=eval_dataloader,
            eval_metric=eval_metric,
        )
        return q_model

    def test_run_mse_one_trial(self):
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion

        tuning_criterion = TuningCriterion(strategy="mse")
        config = PostTrainingQuantConfig(tuning_criterion=tuning_criterion)
        q_model = self._test_tf_model_helper(config, eval_func=lambda mode: 1)
        self.assertIsNotNone(q_model)

    def test_run_mse_max_trials(self):
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion

        tuning_criterion = TuningCriterion(strategy="mse", max_trials=5)
        accuracy_criterion = AccuracyCriterion(tolerable_loss=-0.01)
        config = PostTrainingQuantConfig(tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion)
        q_model = self._test_tf_model_helper(config, eval_func=lambda mode: 1)
        self.assertIsNone(q_model)

    def test_run_rmse_metric(self):
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.metric import METRICS
        from neural_compressor.quantization import fit

        tuning_criterion = TuningCriterion(strategy="mse", max_trials=5)
        accuracy_criterion = AccuracyCriterion(tolerable_loss=-0.9)
        config = PostTrainingQuantConfig(tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion)
        metric = {"topk": 1, "MSE": {"compare_label": False}}

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)

        q_model = self._test_tf_model_helper(config, eval_dataloader=dataloader, eval_metric=metric)
        self.assertIsNone(q_model)

    def _test_ort_model_helper(self, config, eval_func):
        from neural_compressor.data import DATALOADERS
        from neural_compressor.quantization import fit

        # dataset and dataloader
        ds = dataset()
        dataloader = DATALOADERS["onnxrt_qlinearops"](ds)

        q_model = fit(model="mb_v2.onnx", conf=config, calib_dataloader=dataloader, eval_func=eval_func)
        return q_model

    def test_ox_mse(self):
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion

        tuning_criterion = TuningCriterion(strategy="mse", max_trials=5)
        accuracy_criterion = AccuracyCriterion(tolerable_loss=-0.01)
        config = PostTrainingQuantConfig(tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion)
        q_model = self._test_ort_model_helper(config, lambda mode: 1)
        self.assertIsNone(q_model)


if __name__ == "__main__":
    unittest.main()
