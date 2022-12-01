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

import logging
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)

## Required parameters
flags.DEFINE_string(
    'input_model', None, 'Run inference with specified keras model.')

flags.DEFINE_string(
    'output_model', None, 'The output quantized model.')

flags.DEFINE_string(
    'mode', 'performance', 'define benchmark mode for accuracy or performance')

flags.DEFINE_bool(
    'tune', False, 'whether to tune the model')

flags.DEFINE_bool(
    'benchmark', False, 'whether to benchmark the model')

flags.DEFINE_string(
    'config', 'resnet50.yaml', 'yaml configuration of the model')

flags.DEFINE_string(
    'train_data', None, 'location of training dataset on tfrecord format')

flags.DEFINE_string(
    'val_data', None, 'location of validation dataset on tfrecord format')

from neural_compressor.data.transforms.imagenet_transform import (
    TensorflowResizeCropImagenetTransform,
)
from neural_compressor.experimental.data.datasets.dataset import TensorflowImageRecord
from neural_compressor.experimental.data.transforms.transform import ComposeTransform

def prepare_data():
    """Load the dataset of MNIST.

    Returns:
        train (tuple): The images and labels for training.
        test (tuple): The images and labels for testing.
    """
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)

(train_images, train_labels), (test_images, test_labels) = prepare_data()

def evaluate(model, measurer=None):
    """Custom evaluate function to inference the model for specified metric on validation dataset.

    Args:
        model (tf.keras.Model): The input model will be the class of tf.keras.Model.
        measurer (object, optional): for duration benchmark measurement.

    Returns:
        accuracy (float): evaluation result, the larger is better.

    """
    model.compile(
        optimizer="sgd",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    if measurer:
        measurer.start()
    _, q_aware_model_accuracy = model.evaluate(test_images, test_labels)
    if measurer:
        measurer.end()

    return q_aware_model_accuracy

def q_func(model):
    """Custom train function to applying QAT to the fp32 model by calling the ModelConversion class.
    The quantized model will be fine-tuned to compensate the loss brought by quantization.

    Args:
        model (neural_compressor.TensorflowQATModel): The input model will be the INC inner class. Use
                                                      model.model to get a model of tf.keras.Model class.

    Returns:
        q_aware_model (tf.keras.Model): The quantized model by applying QAT.
    """
    # get converted qat model
    from neural_compressor.experimental import ModelConversion
    conversion = ModelConversion()
    q_aware_model = conversion.fit(model.model)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    q_aware_model.summary()

    train_images_subset = train_images[0:1000] # out of 60000
    train_labels_subset = train_labels[0:1000]

    q_aware_model.fit(train_images_subset, train_labels_subset,
                    batch_size=500, epochs=1, validation_split=0.1)

    _, q_aware_model_accuracy = q_aware_model.evaluate(
    test_images, test_labels, verbose=0)

    print('Quant test accuracy:', q_aware_model_accuracy)
    q_aware_model.save("trained_qat_model")
    return q_aware_model

def main():
    logger.info('start quantizating the model...')
    from neural_compressor.experimental import Quantization
    from neural_compressor.experimental import common
    quantizer = Quantization(FLAGS.config)
    quantizer.eval_func = evaluate
    quantizer.q_func = q_func
    quantizer.model = common.Model(FLAGS.input_model)
    q_model = quantizer.fit()
    q_model.save(FLAGS.output_model)

if __name__ == "__main__":
    main()
