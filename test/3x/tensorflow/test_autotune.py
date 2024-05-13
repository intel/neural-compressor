import math
import shutil
import unittest
from functools import wraps
from typing import Callable, Dict, List, Optional, Union
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from tensorflow import keras

from neural_compressor.common import logger
from neural_compressor.common.base_tuning import Evaluator, TuningConfig
from neural_compressor.tensorflow.quantization import SmoothQuantConfig, StaticQuantConfig, autotune
from neural_compressor.tensorflow.utils import version1_gte_version2


def _create_evaluator_for_eval_fns(eval_fns: Optional[Union[Callable, Dict, List[Dict]]] = None) -> Evaluator:
    evaluator = Evaluator()
    evaluator.set_eval_fn_registry(eval_fns)
    return evaluator


def build_model():
    # Load MNIST dataset
    mnist = keras.datasets.mnist

    # 60000 images in train and 10000 images in test, but we don't need so much for ut
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, train_labels = train_images[:1000], train_labels[:1000]
    test_images, test_labels = test_images[:200], test_labels[:200]

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model architecture.
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(28, 28)),
            keras.layers.Reshape(target_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10),
        ]
    )
    # Train the digit classification model
    model.compile(
        optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )

    model.fit(
        train_images,
        train_labels,
        epochs=1,
        validation_split=0.1,
    )

    _, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)

    print("Baseline test accuracy:", baseline_model_accuracy)
    if version1_gte_version2(tf.__version__, "2.16.1"):
        tf.saved_model.save(model, "baseline_model")
    else:
        model.save("baseline_model")


class Dataset(object):
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images, train_labels = train_images[:1000], train_labels[:1000]
        test_images, test_labels = test_images[:200], test_labels[:200]
        # Normalize the input image so that each pixel value is between 0 to 1.
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0
        self.train_labels = train_labels
        self.test_labels = test_labels

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        return self.test_images[idx], self.test_labels[idx]


class MyDataloader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = math.ceil(len(dataset) / self.batch_size)

    def __iter__(self):
        for _, (images, labels) in enumerate(self.dataset):
            images = np.expand_dims(images, axis=0)
            labels = np.expand_dims(labels, axis=0)
            yield (images, labels)

    def __len__(self):
        return self.length


class TestAutoTune(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_model()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("baseline_model", ignore_errors=True)

    def setUp(self):
        # print the test name
        logger.info(f"Running TestAutoTune test: {self.id()}")

    def test_static_quant_auto_tune(self):
        acc_data = iter([1.0, 0.8, 0.99, 1.0, 0.99, 0.99])

        def eval_acc_fn(model) -> float:
            return next(acc_data)

        perf_data = iter([1.0, 0.9, 0.99])

        def eval_perf_fn(model) -> float:
            return next(perf_data)

        calib_dataloader = MyDataloader(dataset=Dataset())
        custom_tune_config = TuningConfig(
            config_set=[
                StaticQuantConfig(weight_sym=True, act_sym=True),
                StaticQuantConfig(weight_sym=False, act_sym=False),
            ]
        )
        best_model = autotune(
            model="baseline_model",
            tune_config=custom_tune_config,
            eval_fn=eval_acc_fn,
            calib_dataloader=calib_dataloader,
        )
        self.assertIsNotNone(best_model)

    def test_sq_auto_tune(self):
        acc_data = iter([1.0, 0.8, 0.6, 1.0, 0.99, 0.9])

        def eval_acc_fn(model) -> float:
            return next(acc_data)

        perf_data = iter([1.0, 0.99, 0.99])

        def eval_perf_fn(model) -> float:
            return next(perf_data)

        eval_fns = [
            {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
            {
                "eval_fn": eval_perf_fn,
                "weight": 0.5,
            },
        ]

        evaluator = _create_evaluator_for_eval_fns(eval_fns)

        def eval_fn_wrapper(model):
            result = evaluator.evaluate(model)
            return result

        calib_dataloader = MyDataloader(dataset=Dataset())
        custom_tune_config = TuningConfig(config_set=[SmoothQuantConfig(alpha=0.5), SmoothQuantConfig(alpha=0.6)])
        best_model = autotune(
            model="baseline_model",
            tune_config=custom_tune_config,
            eval_fn=eval_acc_fn,
            calib_dataloader=calib_dataloader,
        )
        self.assertIsNone(best_model)

        custom_tune_config = TuningConfig(config_set=[SmoothQuantConfig(alpha=[0.5, 0.6])])
        best_model = autotune(
            model="baseline_model",
            tune_config=custom_tune_config,
            eval_fn=eval_fn_wrapper,
            calib_dataloader=calib_dataloader,
        )
        self.assertEqual(len(evaluator.eval_fn_registry), 2)
        self.assertIsNotNone(best_model)

        op_names = [i.name for i in best_model.graph_def.node if i.op == "MatMul" and "_mul" in i.input[0]]
        self.assertTrue(len(op_names) > 0)


if __name__ == "__main__":
    unittest.main()
