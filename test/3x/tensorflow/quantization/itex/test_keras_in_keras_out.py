#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time
import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

from neural_compressor.utils import logger

test_mode = "accuracy"


def build_model():
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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
    model.save("baseline_model")


def build_dataset():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def eval_func(model):
    x_train, y_train, x_test, y_test = build_dataset()
    start = time.time()
    model.compile(metrics=["accuracy"], run_eagerly=False)
    score = model.evaluate(x_test, y_test)
    end = time.time()

    if test_mode == "performance":
        latency = end - start
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} data/sec".format(1.0 / latency))
    return score[1]


class Dataset(object):
    def __init__(self, batch_size=100):
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize the input image so that each pixel value is between 0 to 1.
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0
        self.train_labels = train_labels
        self.test_labels = test_labels

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        return self.test_images[idx], self.test_labels[idx]


class TestKerasInKerasOut(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.environ["ITEX_ONEDNN_GRAPH"] = "1"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("baseline_model", ignore_errors=True)
        shutil.rmtree("itex_qdq_keras_model", ignore_errors=True)

    def test_keras_in_keras_out(self):
        logger.info("Run test_keras_in_keras_out case...")
        global test_mode
        test_mode = "accuracy"
        build_model()

        from neural_compressor import set_random_seed
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.data.dataloaders.dataloader import DataLoader
        from neural_compressor.quantization import fit

        set_random_seed(9527)
        config = PostTrainingQuantConfig(backend="itex")
        logger.info("=================Run Quantization...")
        q_model = fit(
            keras.models.load_model("./baseline_model"),
            conf=config,
            calib_dataloader=DataLoader(framework="tensorflow", dataset=Dataset()),
            eval_func=eval_func,
        )
        q_model.save("itex_qdq_keras_model")
        self.assertEqual(q_model.framework(), "keras")

        framework_config = {"framework": "keras", "approach": "post_training_static_quant"}
        q_model.q_config = framework_config
        self.assertEqual(q_model.q_config["framework"], "keras")
        self.assertEqual(q_model.graph_info, None)
        self.assertEqual(q_model.framework(), "keras")
        self.assertEqual(isinstance(q_model.model, tf.keras.Model), True)

        model = keras.models.load_model("./itex_qdq_keras_model")
        model.summary()
        found_quantize = False
        found_dequantize = False
        for layer in model.layers:
            if "quantize" in layer.name:
                found_quantize = True
            if "dequantize" in layer.name:
                found_dequantize = True
        self.assertEqual(found_quantize, True)
        self.assertEqual(found_dequantize, True)

        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig

        conf = BenchmarkConfig(backend="itex", iteration=100, cores_per_instance=1, num_of_instance=1)
        logger.info("=================Run BenchMark...")
        test_mode = "performance"
        fit(model, conf, b_func=eval_func)


if __name__ == "__main__":
    unittest.main()
