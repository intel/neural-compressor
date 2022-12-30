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
from neural_compressor.data.transforms.transform import ComposeTransform
from neural_compressor.data.datasets.dataset import TensorflowImageRecord
from neural_compressor.data.transforms.imagenet_transform import LabelShift
from neural_compressor.data.transforms.imagenet_transform import TensorflowResizeCropImagenetTransform

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)

# Required parameters
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
    'dataset_location', None, 'location of the dataset on tfrecord format')

flags.DEFINE_integer(
    'batch_size', 32, 'batch_size')

flags.DEFINE_integer(
    'iters', -1, 'iteration')


def prepare_data(root):
    """
    Parse the input tf_record data.

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
                height=224, width=224, mean_value=[123.68, 116.78, 103.94])
        ]))

    data = np.array(list(dataset.map(lambda x, y: x)))
    data = tf.keras.applications.resnet.preprocess_input(data)
    label = np.array(list(dataset.map(lambda x, y: y))).squeeze(1)

    if len(data) > 10000:
        data = data[:10000]
        label = label[:10000]

    for idx, i in enumerate(label):
        label[idx] = i-1

    return data, label


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
    output_tensor = model.output_tensor if len(
        model.output_tensor) > 1 else model.output_tensor[0]
    postprocess = LabelShift(label_shift=1)
    metric = TensorflowTopK(k=1)

    def eval_func(dataloader):
        latency_list = []
        for idx, (inputs, labels) in enumerate(dataloader):
            # dataloader should keep the order and len of inputs same with input_tensor
            inputs = np.array([inputs])
            assert len(input_tensor) == len(inputs), \
                'inputs len must equal with input_tensor'
            feed_dict = dict(zip(input_tensor, inputs))

            start = time.time()
            predictions = model.sess.run(output_tensor, feed_dict)
            end = time.time()

            predictions, labels = postprocess((predictions, labels))
            metric.update(predictions, labels)
            latency_list.append(end-start)
            if idx + 1 == FLAGS.iters and FLAGS.mode == 'performance':
                break
        latency = np.array(latency_list).mean() / FLAGS.batch_size
        return latency

    from neural_compressor.experimental.data.dataloaders.default_dataloader import DefaultDataLoader
    dataset = TensorflowImageRecord(root=FLAGS.dataset_location, transform=ComposeTransform(transform_list=[
            TensorflowResizeCropImagenetTransform(height=224, width=224, mean_value=[123.68, 116.78, 103.94])]))
    dataloader = DefaultDataLoader(dataset, batch_size=FLAGS.batch_size)
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
        q_aware_model.compile(
            optimizer='sgd',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        q_aware_model.summary()
        x_train, y_train = prepare_data(FLAGS.dataset_location)
        q_aware_model.fit(x_train, y_train, batch_size=64, epochs=3)

        compression_manager.callbacks.on_train_end()
        compression_manager.save(FLAGS.output_model)

    if FLAGS.benchmark:
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        from neural_compressor.model.tensorflow_model import TensorflowQATModel
        assert FLAGS.mode == 'performance' or FLAGS.mode == 'accuracy', \
        "Benchmark only supports performance or accuracy mode."

        model = TensorflowQATModel(FLAGS.input_model).frozen_graph_def
        if FLAGS.mode == 'performance':
            conf = BenchmarkConfig(cores_per_instance=4, num_of_instance=7)
            fit(model, conf, b_func=evaluate)
        elif FLAGS.mode == 'accuracy':
            accuracy = evaluate(model)
            print('Batch size = %d' % FLAGS.batch_size)
            print("Accuracy: %.5f" % accuracy)


if __name__ == "__main__":
    main()
