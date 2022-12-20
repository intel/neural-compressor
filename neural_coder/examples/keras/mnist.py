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
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
from torch.utils.data import DataLoader

num_classes = 10

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
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

class Dataset():
    def __init__(self, ):
         _, _ , self.inputs, self.labels = build_dataset()

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        assert len(self.inputs) == len(self.labels), 'inputs should have equal len with labels'
        return len(self.inputs)

def build_model(x_train, y_train, x_test, y_test):
    if os.path.exists('fp32_model'):
        model = keras.models.load_model('fp32_model')
        return model
    # Model / data parameters
    input_shape = (28, 28, 1)
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    
    batch_size = 128
    epochs = 1
    
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"], run_eagerly=True)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.summary()
    if not os.path.exists('fp32_model'):
        model.save('fp32_model')
    return model

def eval_func(model):
    x_train, y_train, x_test, y_test = build_dataset()
    model.compile(metrics=["accuracy"], run_eagerly=False)
    score = model.evaluate(x_test, y_test)
    return score[1]

def main():
    x_train, y_train, x_test, y_test = build_dataset()
    model = build_model(x_train, y_train, x_test, y_test)
    calib_dataloader = DataLoader(Dataset(), batch_size=10)

if __name__ == '__main__':
    main()
