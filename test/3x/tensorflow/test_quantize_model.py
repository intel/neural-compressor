#!/usr/bin/env python
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
import shutil
import time
import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

from neural_compressor.common import logger
from neural_compressor.tensorflow.utils import version1_gte_version2


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
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu", name="conv2d"),
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
        model.export("baseline_model")
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


def evaluate(model):
    input_tensor = model.input_tensor
    output_tensor = model.output_tensor if len(model.output_tensor) > 1 else model.output_tensor[0]

    iteration = -1
    calib_dataloader = MyDataloader(dataset=Dataset())
    for idx, (inputs, labels) in enumerate(calib_dataloader):
        # dataloader should keep the order and len of inputs same with input_tensor
        inputs = np.array([inputs])
        feed_dict = dict(zip(input_tensor, inputs))

        start = time.time()
        predictions = model.sess.run(output_tensor, feed_dict)
        end = time.time()

        if idx + 1 == iteration:
            break


class TestQuantizeModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_model()
        self.fp32_model_path = "baseline_model"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.fp32_model_path, ignore_errors=True)

    def test_calib_func(self):
        logger.info("Run test_calib_func case...")

        from neural_compressor.common import set_random_seed
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model

        set_random_seed(9527)
        quant_config = StaticQuantConfig()
        q_model = quantize_model(self.fp32_model_path, quant_config, calib_func=evaluate)
        quantized = False
        for node in q_model.graph_def.node:
            if "Quantized" in node.op:
                quantized = True
                break

        self.assertEqual(quantized, True)


if __name__ == "__main__":
    unittest.main()
