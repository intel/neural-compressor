# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

import math
import os
import shutil
import unittest

import keras
import numpy as np
import tensorflow as tf

from neural_compressor.common import Logger
from neural_compressor.tensorflow.utils import version1_gte_version2

logger = Logger().get_logger()


def build_model1():
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
            keras.layers.DepthwiseConv2D(3, 3, activation="relu", name="conv2d"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, name="dense"),
        ]
    )
    # Train the digit classification model
    model.compile(
        optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
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
        model.save("baseline_model1.keras")
    else:
        model.save("baseline_model1")


def build_model2():
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
            keras.layers.SeparableConv2D(3, 3, activation="relu"),
            keras.layers.AveragePooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, name="dense"),
        ]
    )
    # Train the digit classification model
    model.compile(
        optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
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
        model.save("baseline_model2.keras")
    else:
        model.save("baseline_model2")


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


class TestTF3xNewApi(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_model1()
        build_model2()
        os.environ["ITEX_ONEDNN_GRAPH"] = "1"
        self.fp32_model_path1 = (
            "baseline_model1.keras" if version1_gte_version2(tf.__version__, "2.16.1") else "baseline_model1"
        )
        self.fp32_model_path2 = (
            "baseline_model2.keras" if version1_gte_version2(tf.__version__, "2.16.1") else "baseline_model2"
        )

    @classmethod
    def tearDownClass(self):
        if self.fp32_model_path1.endswith(".keras"):
            os.remove(self.fp32_model_path1)
            os.remove(self.fp32_model_path2)
        else:
            shutil.rmtree(self.fp32_model_path1, ignore_errors=True)
            shutil.rmtree(self.fp32_model_path2, ignore_errors=True)
        os.environ["ITEX_ONEDNN_GRAPH"] = "0"

    def test_depthwise_conv2d(self):
        logger.info("test_static_quant_from_dict_default")
        from neural_compressor.tensorflow import quantize_model
        from neural_compressor.tensorflow.keras import get_default_static_quant_config

        calib_dataloader = MyDataloader(dataset=Dataset())
        fp32_model = keras.models.load_model(self.fp32_model_path1)
        qmodel = quantize_model(fp32_model, get_default_static_quant_config(), calib_dataloader)
        self.assertIsNotNone(qmodel)

        for layer in qmodel.layers:
            if layer.name == "conv2d":
                self.assertEqual(layer.__class__.__name__, "QDepthwiseConv2D")
                break

    def test_seprable_conv2d(self):
        logger.info("test_static_quant_from_dict_default")
        from neural_compressor.tensorflow import quantize_model
        from neural_compressor.tensorflow.keras import get_default_static_quant_config

        calib_dataloader = MyDataloader(dataset=Dataset())
        fp32_model = keras.models.load_model(self.fp32_model_path2)
        qmodel = quantize_model(fp32_model, get_default_static_quant_config(), calib_dataloader)
        self.assertIsNotNone(qmodel)

        for layer in qmodel.layers:
            if layer.name == "conv2d":
                self.assertEqual(layer.__class__.__name__, "QSeparableConv2D")
                break


if __name__ == "__main__":
    unittest.main()
