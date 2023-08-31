"""Tests for quantization."""
import os
import shutil
import unittest

import numpy as np

if os.getenv("SIGOPT_API_TOKEN") is None or os.getenv("SIGOPT_PROJECT_ID") is None:
    CONDITION = True
else:
    CONDITION = False


def build_fake_model():
    import tensorflow as tf

    try:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1, 3, 3, 1), name="x")
            y = tf.constant(np.random.random((2, 2, 1, 1)).astype(np.float32), name="y")
            z = tf.constant(np.random.random((1, 1, 1, 1)).astype(np.float32), name="z")
            op = tf.nn.conv2d(input=tf.nn.relu(x), filters=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")
            op2 = tf.nn.conv2d(
                input=tf.nn.relu(op), filters=z, strides=[1, 1, 1, 1], padding="VALID", name="op2_to_store"
            )

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
            op2 = tf.nn.conv2d(input=op, filters=z, strides=[1, 1, 1, 1], padding="VALID", name="op2_to_store")

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ["op2_to_store"]
            )

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    return graph


class TestSigoptTuningStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        sigopt_api_token = os.getenv("SIGOPT_API_TOKEN")
        sigopt_project_id = os.getenv("SIGOPT_PROJECT_ID")
        self.constant_graph = build_fake_model()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("saved", ignore_errors=True)

    @unittest.skipIf(CONDITION, "missing the env variables 'SIGOPT_API_TOKEN' or 'SIGOPT_PROJECT_ID'")
    def test_run_sigopt_one_trial_new_api(self):
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)

        # tuning and accuracy criterion
        accuracy_criterion = AccuracyCriterion(criterion="relative")
        strategy_kwargs = {
            "sigopt_api_token": "sigopt_api_token_test",
            "sigopt_project_id": "sigopt_project_id_test",
            "sigopt_experiment_name": "nc-tune",
        }
        tuning_criterion = TuningCriterion(strategy="sigopt", strategy_kwargs=strategy_kwargs, max_trials=3)
        conf = PostTrainingQuantConfig(
            quant_level=1, approach="static", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion
        )
        self.assertEqual(conf.tuning_criterion.strategy_kwargs, strategy_kwargs)

        def fake_eval(model):
            return 1

        q_model = fit(model=self.constant_graph, conf=conf, calib_dataloader=dataloader, eval_func=fake_eval)

    def test_run_sigopt_one_trial_fake_token(self):
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)

        # tuning and accuracy criterion
        accuracy_criterion = AccuracyCriterion(criterion="relative")
        strategy_kwargs = {
            "sigopt_api_token": "sigopt_api_token_test",
            "sigopt_project_id": "sigopt_project_id_test",
            "sigopt_experiment_name": "nc-tune",
        }
        tuning_criterion = TuningCriterion(strategy="sigopt", strategy_kwargs=strategy_kwargs, max_trials=3)
        conf = PostTrainingQuantConfig(
            quant_level=1, approach="static", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion
        )
        self.assertEqual(conf.tuning_criterion.strategy_kwargs, strategy_kwargs)

        def fake_eval(model):
            return 1

        q_model = fit(model=self.constant_graph, conf=conf, calib_dataloader=dataloader, eval_func=fake_eval)


if __name__ == "__main__":
    unittest.main()
