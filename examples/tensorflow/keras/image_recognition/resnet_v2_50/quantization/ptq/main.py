#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import time

import numpy as np
import tensorflow as tf

from neural_compressor.utils import logger
from data_process import (
    ImageRecordDataset, 
    ComposeTransform, 
    BilinearImagenetTransform, 
    TFDataLoader,
    TopKMetric,
    LabelShift
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

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
    'calib_data', None, 'location of calibration dataset')

flags.DEFINE_string(
    'eval_data', None, 'location of evaluate dataset')

flags.DEFINE_integer('batch_size', 32, 'batch_size')

flags.DEFINE_integer(
    'iters', 100, 'maximum iteration when evaluating performance')

height = width = 224
eval_dataset = ImageRecordDataset(root=FLAGS.eval_data, transform=ComposeTransform(transform_list= \
                 [BilinearImagenetTransform(height=height, width=width)]))

eval_dataloader = TFDataLoader(dataset=eval_dataset, batch_size=FLAGS.batch_size)

if FLAGS.calib_data:
    calib_dataset = ImageRecordDataset(root=FLAGS.calib_data, transform= \
        ComposeTransform(transform_list= [BilinearImagenetTransform(height=height, width=width)]))
    calib_dataloader = TFDataLoader(dataset=calib_dataset, batch_size=10)

def evaluate(model):
    """
    Custom evaluate function to inference the model for specified metric on validation dataset.

    Args:
       model (tf.keras.Model): The input model will be the objection of tf.keras.Model.

    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    latency_list = []
    metric = TopKMetric()
    postprocess = LabelShift(label_shift=1)

    def eval_func(dataloader, metric):
        warmup = 5
        iteration = None
        if FLAGS.benchmark and FLAGS.mode == 'performance':
            iteration = FLAGS.iters
        for idx, (inputs, labels) in enumerate(dataloader):
            start = time.time()
            predictions = model.predict_on_batch(inputs)
            end = time.time()
            latency_list.append(end - start)
            predictions, labels = postprocess((predictions, labels))
            metric.update(predictions, labels)
            if iteration and idx >= iteration:
                break
        latency = np.array(latency_list[warmup:]).mean() / eval_dataloader.batch_size
        return latency

    latency = eval_func(eval_dataloader, metric)
    if FLAGS.benchmark:
        logger.info("\n{} mode benchmark result:".format(FLAGS.mode))
        for i, res in enumerate(latency_list):
            logger.debug("Iteration {} result {}:".format(i, res))
    if FLAGS.benchmark and FLAGS.mode == 'performance':
        logger.info("Batch size = {}".format(eval_dataloader.batch_size))
        logger.info("Latency: {:.3f} ms".format(latency * 1000))
        logger.info("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc

def main(_):
    if FLAGS.tune:
        from neural_compressor.common import set_random_seed
        from neural_compressor.tensorflow import quantize_model
        from neural_compressor.tensorflow.keras import StaticQuantConfig

        set_random_seed(9527)
        quant_config = StaticQuantConfig()
        q_model = quantize_model(FLAGS.input_model, quant_config, calib_dataloader)
        q_model.save(FLAGS.output_model)
        logger.info("Save quantized model to {}.".format(FLAGS.output_model))

    if FLAGS.benchmark:
        from neural_compressor.tensorflow import Model

        inc_model = Model(FLAGS.input_model)
        if FLAGS.mode == 'performance':
            evaluate(inc_model.model)
        else:
            accuracy = evaluate(inc_model.model)
            logger.info('Batch size = %d' % FLAGS.batch_size)
            logger.info("Accuracy: %.5f" % accuracy)

if __name__ == "__main__":
    tf.compat.v1.app.run()
