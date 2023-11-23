#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
    model.save("baseline_model")


class Dataset(object):
    def __init__(self, batch_size=100):
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


class TestKeras3xNewApi(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_model()
        os.environ["ITEX_ONEDNN_GRAPH"] = "1"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("baseline_model", ignore_errors=True)
        shutil.rmtree("int8_model", ignore_errors=True)

    def test_static_quant_from_dict_default(self):
        logger.info("test_static_quant_from_dict_default")
        from neural_compressor.tensorflow.utils import TFDataDataLoader
        from neural_compressor.tensorflow import get_default_static_quant_config, quantize_model

        calib_dataloader = TFDataDataLoader(dataset=Dataset())
        fp32_model = keras.models.load_model("./baseline_model")
        qmodel = quantize_model(fp32_model, get_default_static_quant_config(), calib_dataloader)
        self.assertIsNotNone(qmodel)

        for layer in qmodel.layers:
            if layer.name == "dense":
                self.assertEqual(layer.__class__.__name__, "QDense")

    def test_static_quant_from_dict_beginner(self):
        logger.info("test_static_quant_from_dict_beginner")
        from neural_compressor.tensorflow.utils import TFDataDataLoader
        from neural_compressor.tensorflow import quantize_model

        calib_dataloader = TFDataDataLoader(dataset=Dataset())

        quant_config = {
            "static_quant": {
                "global": {
                    "weight_dtype": "int8",
                    "weight_sym": True,
                    "weight_granularity": "per_tensor",
                    "act_dtype": "int8",
                    "act_sym": True,
                    "act_granularity": "per_tensor",
                },
            }
        }
        fp32_model = keras.models.load_model("./baseline_model")
        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

        for layer in qmodel.layers:
            if layer.name == "dense":
                self.assertEqual(layer.__class__.__name__, "QDense")

    def test_static_quant_from_class_default(self):
        logger.info("test_static_quant_from_class_default")
        from neural_compressor.tensorflow.utils import TFDataDataLoader
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model

        calib_dataloader = TFDataDataLoader(dataset=Dataset())
        fp32_model = keras.models.load_model("./baseline_model")
        quant_config = StaticQuantConfig()
        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

        for layer in qmodel.layers:
            if layer.name == "dense":
                self.assertEqual(layer.__class__.__name__, "QDense")

    def test_static_quant_from_class_beginner(self):
        logger.info("test_static_quant_from_class_beginner")
        from neural_compressor.tensorflow.utils import TFDataDataLoader
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model

        calib_dataloader = TFDataDataLoader(dataset=Dataset())
        fp32_model = keras.models.load_model("./baseline_model")
        quant_config = StaticQuantConfig(
            weight_dtype="int8",
            weight_sym=True,
            weight_granularity="per_channel",
            act_dtype="int8",
            act_sym=True,
            act_granularity="per_channel",
        )
        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

        for layer in qmodel.layers:
            if layer.name == "dense":
                self.assertEqual(layer.__class__.__name__, "QDense")

    def test_static_quant_from_dict_advance(self):
        logger.info("test_static_quant_from_dict_advance")
        from neural_compressor.tensorflow.utils import TFDataDataLoader
        from neural_compressor.tensorflow import quantize_model

        calib_dataloader = TFDataDataLoader(dataset=Dataset())
        fp32_model = keras.models.load_model("./baseline_model")
        quant_config = {
            "static_quant": {
                "global": {
                    "weight_dtype": "int8",
                    "weight_sym": True,
                    "weight_granularity": "per_tensor",
                    "act_dtype": "int8",
                    "act_sym": True,
                    "act_granularity": "per_tensor",
                },
                "local": {
                    "dense": {
                        "weight_dtype": "fp32",
                        "act_dtype": "fp32",
                    }
                },
            }
        }
        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

        for layer in qmodel.layers:
            if layer.name == "dense":
                self.assertNotEqual(layer.__class__.__name__, "QDense")

    def test_static_quant_from_class_advance(self):
        logger.info("test_static_quant_from_class_advance")
        from neural_compressor.tensorflow.utils import TFDataDataLoader
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model

        calib_dataloader = TFDataDataLoader(dataset=Dataset())
        quant_config = StaticQuantConfig(
            weight_dtype="int8",
            weight_sym=True,
            weight_granularity="per_channel",
            act_dtype="int8",
            act_sym=True,
            act_granularity="per_channel",
        )
        dense_config = StaticQuantConfig(
            weight_dtype="fp32",
            act_dtype="fp32",
        )
        quant_config.set_local("dense", dense_config)
        # get model and quantize
        fp32_model = keras.models.load_model("./baseline_model")
        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

        for layer in qmodel.layers:
            if layer.name == "dense":
                self.assertNotEqual(layer.__class__.__name__, "QDense")

    def test_config_from_dict(self):
        logger.info("test_config_from_dict")
        from neural_compressor.tensorflow import StaticQuantConfig

        quant_config = {
            "static_quant": {
                "global": {
                    "weight_dtype": "int8",
                    "weight_sym": True,
                    "weight_granularity": "per_tensor",
                    "act_dtype": "int8",
                    "act_sym": True,
                    "act_granularity": "per_tensor",
                },
                "local": {
                    "dense": {
                        "weight_dtype": "fp32",
                        "act_dtype": "fp32",
                    }
                },
            }
        }
        config = StaticQuantConfig.from_dict(quant_config)
        self.assertIsNotNone(config.local_config)

    def test_config_to_dict(self):
        logger.info("test_config_to_dict")
        from neural_compressor.tensorflow import StaticQuantConfig

        quant_config = StaticQuantConfig(
            weight_dtype="int8",
            weight_sym=True,
            weight_granularity="per_channel",
            act_dtype="int8",
            act_sym=True,
            act_granularity="per_channel",
        )
        dense_config = StaticQuantConfig(
            weight_dtype="fp32",
            act_dtype="fp32",
        )
        quant_config.set_local("dense", dense_config)
        config_dict = quant_config.to_dict()
        self.assertIn("global", config_dict)
        self.assertIn("operator_name", config_dict)


if __name__ == "__main__":
    unittest.main()
