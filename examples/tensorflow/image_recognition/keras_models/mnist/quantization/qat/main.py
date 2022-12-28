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

import time
import logging
import numpy as np
import tensorflow as tf

from neural_compressor.metric.metric import TensorflowTopK

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

flags.DEFINE_integer(
    'batch_size', 32, 'batch_size')


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

class Dataloader(object):
    def __init__(self, batch_size=100):
        mnist = tf.keras.datasets.mnist
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

def evaluate(model):
    """Custom evaluate function to estimate the accuracy of the model.

    Args:
        model (tf.Graph_def): The input model graph
        
    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    from neural_compressor.model.model import Model
    model = Model(model)
    input_tensor = model.input_tensor
    output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                        model.output_tensor[0]
    iteration = -1
    if FLAGS.benchmark and FLAGS.mode == 'performance':
        iteration = 100
    metric = TensorflowTopK(k=1)

    def eval_func(dataloader):
        latency_list = []
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = np.array([inputs])
            # dataloader should keep the order and len of inputs same with input_tensor
            assert len(input_tensor) == len(inputs), \
                'inputs len must equal with input_tensor'
            feed_dict = dict(zip(input_tensor, inputs))

            start = time.time()
            predictions = model.sess.run(output_tensor, feed_dict)
            end = time.time()

            metric.update(predictions, labels)
            latency_list.append(end-start)
            if idx + 1 == iteration:
                break
        latency = np.array(latency_list).mean() / FLAGS.batch_size
        return latency

    dataloader = Dataloader(batch_size=FLAGS.batch_size)
    latency = eval_func(dataloader)
    if FLAGS.benchmark and FLAGS.mode == 'performance':
        print("Batch size = {}".format(FLAGS.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc


def main():
    if FLAGS.tune:
        logger.info('start quantizing the model...')
        from neural_compressor import training, QuantizationAwareTrainingConfig
        config = QuantizationAwareTrainingConfig()
        compression_manager = training.prepare_compression(FLAGS.input_model, config)
        compression_manager.callbacks.on_train_begin()

        q_aware_model = compression_manager.model.model

        q_aware_model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        q_aware_model.summary()
        train_images_subset = train_images[0:1000]
        train_labels_subset = train_labels[0:1000]
        q_aware_model.fit(train_images_subset, train_labels_subset,
                        batch_size=500, epochs=1, validation_split=0.1)
        _, q_aware_model_accuracy = q_aware_model.evaluate(
                                        test_images, test_labels, verbose=0)
        print('Quant test accuracy:', q_aware_model_accuracy)

        compression_manager.callbacks.on_train_end()
        compression_manager.save(FLAGS.output_model)

    if FLAGS.benchmark:
        from neural_compressor.benchmark import fit
        from neural_compressor.model.model import Model
        from neural_compressor.config import BenchmarkConfig
        assert FLAGS.mode == 'performance' or FLAGS.mode == 'accuracy', \
        "Benchmark only supports performance or accuracy mode."

        model = Model(FLAGS.input_model).graph_def
        if FLAGS.mode == 'performance':
            conf = BenchmarkConfig(cores_per_instance=4, num_of_instance=7)
            fit(model, conf, b_func=evaluate)
        elif FLAGS.mode == 'accuracy':
            accuracy = evaluate(model)
            print('Batch size = %d' % FLAGS.batch_size)
            print("Accuracy: %.5f" % accuracy)

if __name__ == "__main__":
    main()
