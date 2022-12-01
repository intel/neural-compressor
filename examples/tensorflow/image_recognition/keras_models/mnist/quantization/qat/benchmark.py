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

import tensorflow as tf
from tensorflow import keras
import numpy as np

class dataloader(object):
    def __init__(self, batch_size=100):
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        # Normalize the input image so that each pixel value is between 0 to 1.
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0 
        self.train_labels = train_labels
        self.test_labels = test_labels

        self.batch_size = batch_size
        self.i = 0

    def __iter__(self):
        while self.i < len(self.test_images):
            yield self.test_images[self.i: self.i + self.batch_size], self.test_labels[self.i: self.i + self.batch_size]
            self.i = self.i + self.batch_size

from neural_compressor.experimental import Benchmark, common
evaluator = Benchmark('mnist.yaml')
evaluator.model = common.Model('quantized_model')
evaluator.b_dataloader = dataloader()
evaluator('accuracy')
