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

from neural_compressor.data.dataloaders.dataloader import DataLoader


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
        # 60000 images in train and 10000 images in test, but we don't need so much for ut
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
        return self.test_images[idx].astype(np.float32), self.test_labels[idx]


class TestBigSavedModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_model()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("baseline_model", ignore_errors=True)
        shutil.rmtree("int8_model", ignore_errors=True)

    def test_newapi_sq_big_saved_model(self):
        def weight_name_mapping(name):
            """The function that maps name from AutoTrackable variables to graph nodes"""
            name = name.replace('dense', 'StatefulPartitionedCall/sequential/dense/MatMul')
            name = name.replace('conv2d', 'StatefulPartitionedCall/sequential/conv2d/Conv2D')
            name = name.replace('kernel:0', 'ReadVariableOp')
            return name

        from neural_compressor import Model
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig

        model = Model('baseline_model' , modelType='llm_saved_model')
        model.weight_name_mapping = weight_name_mapping

        output_node_names = model.output_node_names
        self.assertEqual(output_node_names, ['Identity'])

        calib_dataloader=DataLoader(framework="tensorflow", dataset=Dataset())
        recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": 0.6}}
        op_name_dict = {
            'StatefulPartitionedCall/sequential/conv2d/Conv2D':{"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}
            }
        config = PostTrainingQuantConfig(
            quant_level=1,
            recipes=recipes,
            op_name_dict=op_name_dict,
            calibration_sampling_size=[500],
        )
        model.weight_name_mapping = weight_name_mapping
        q_model = quantization.fit( model,
                                    config,
                                    calib_dataloader=calib_dataloader)
        q_model.save("int8_model")
        quant_count = 0
        for i in q_model.graph_def.node:
            if i.op == "QuantizeV2":
                quant_count += 1

        self.assertEqual(quant_count, 3)

if __name__ == "__main__":
    unittest.main()