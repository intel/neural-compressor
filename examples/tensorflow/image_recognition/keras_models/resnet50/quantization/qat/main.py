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

def prepare_data(root):
    """Parse the input tf_record data.

    Args:
        root (string): The path to tfrecord files.

    Returns:
        data (float): The images that can be used for training or evaluation.
        label (float): The labels corresponding to the images.
    """
    dataset = TensorflowImageRecord(
        root=root,
        transform=ComposeTransform(transform_list=[
            TensorflowResizeCropImagenetTransform(
                height=224, width=224)
        ]))
    data = np.array(list(dataset.map(lambda x, y: x)))
    data = tf.keras.applications.resnet.preprocess_input(data)
    label = np.array(list(dataset.map(lambda x, y: y))).squeeze(1)
    for idx, i in enumerate(label):
        label[idx] = i-1
    return data, label

# Load the image data.
x_train, y_train = prepare_data(FLAGS.train_data)
x_val, y_val = prepare_data(FLAGS.val_data)

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
    _, q_aware_model_accuracy = model.evaluate(x_val, y_val)
    if measurer:
        measurer.end()

    return q_aware_model_accuracy

def q_func(model):
    """Custom train function to applying QAT to the fp32 model by calling the ModelConversion class.
    The quantized model will be fine-tuned to compensate the loss brought by quantization.

    Args:
        model (neural_compressor.model.model.TensorflowQATModel): The input model will be the INC inner class. Use
                                                                  model.model to get a model of tf.keras.Model class.

    Returns:
        q_aware_model (tf.keras.Model): The quantized model by applying QAT.
    """

    from neural_compressor.experimental import ModelConversion
    conversion = ModelConversion()
    q_aware_model = conversion.fit(model.model)

    q_aware_model.compile(
        optimizer='sgd',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    q_aware_model.summary()
    q_aware_model.fit(x_train,
                        y_train,
                        batch_size=64,
                        epochs=1)

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
