#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

flags.DEFINE_integer(
    'batch_size', 32, 'batch_size of evaluation')

flags.DEFINE_integer(
    'iters', 100, 'maximum iteration when evaluating performance')

from neural_compressor import Metric

def evaluate(model):
    """Custom evaluate function to inference the model for specified metric on validation dataset.

    Args:
        model (tf.keras.Model): The input model will be the objection of tf.keras.Model.

    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    from neural_compressor import METRICS
    metrics = METRICS('tensorflow')
    metric = metrics['topk']()

    def eval_func(data_loader, metric):
        warmup = 5
        iteration = None
        latency_list = []
        if FLAGS.benchmark and FLAGS.mode == 'performance':
            iteration = FLAGS.iters
        for idx, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            predictions = model.predict_on_batch(inputs)
            end = time.time()
            metric.update(predictions, labels)
            latency_list.append(end - start)
            if iteration and idx >= iteration:
                break
        latency = np.array(latency_list[warmup:]).mean() / dataloader.batch_size
        return latency

    from neural_compressor.utils.create_obj_from_config import create_dataloader
    dataloader_args = {
        'batch_size': FLAGS.batch_size,
        'dataset': {"FashionMNIST": {'root':FLAGS.eval_data}},
        'transform': {'Rescale': {}},
        'filter': None
    }
    dataloader = create_dataloader('tensorflow', dataloader_args)
    latency = eval_func(dataloader, metric)
    if FLAGS.benchmark and FLAGS.mode == 'performance':
        print("Batch size = {}".format(dataloader.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc

def main(_):
    from neural_compressor import set_random_seed
    set_random_seed(9527)
    if FLAGS.tune:
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig
        conf = PostTrainingQuantConfig(backend='itex',
                calibration_sampling_size=[50, 100])
        from neural_compressor.utils.create_obj_from_config import create_dataloader
        calib_dataloader_args = {
            'batch_size': FLAGS.batch_size,
            'dataset': {"FashionMNIST": {'root':FLAGS.eval_data}},
            'transform': {'Rescale': {}},
            'filter': None
        }
        calib_dataloader = create_dataloader('tensorflow', calib_dataloader_args)
        q_model = quantization.fit(FLAGS.input_model, conf=conf, calib_dataloader=calib_dataloader,
                    eval_func=evaluate)
        q_model.save(FLAGS.output_model)

    if FLAGS.benchmark:
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        if FLAGS.mode == 'performance':
            conf = BenchmarkConfig(backend='itex',
                    warmup=10, iteration=100, cores_per_instance=4, num_of_instance=1)
            fit(FLAGS.input_model, conf, b_func=evaluate)
        else:
            from neural_compressor.model import Model
            model = Model(FLAGS.input_model, backend='itex').model
            accuracy = evaluate(model)
            print('Batch size = %d' % FLAGS.batch_size)
            print("Accuracy: %.5f" % accuracy)

if __name__ == "__main__":
    tf.compat.v1.app.run()
