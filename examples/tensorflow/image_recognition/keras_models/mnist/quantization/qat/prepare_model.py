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

import argparse
import tensorflow as tf
from tensorflow import keras

def train_func():
  # Load MNIST dataset
  mnist = keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  # Normalize the input image so that each pixel value is between 0 to 1.
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  # Define the model architecture.
  model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28)),
    keras.layers.Reshape(target_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
  ])

  # Train the digit classification model
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  model.fit(
    train_images,
    train_labels,
    epochs=1,
    validation_split=0.1,
  )

  _, baseline_model_accuracy = model.evaluate(
      test_images, test_labels, verbose=0)

  print('Baseline test accuracy:', baseline_model_accuracy)

  return model

def get_mnist_model(saved_path):
    assert saved_path is not None, "save path should not be None"
    model = train_func()
    model.save(saved_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export pretained keras model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_model',
                        type=str,
                        help='path to exported model file')

    args = parser.parse_args()
    get_mnist_model(args.output_model)