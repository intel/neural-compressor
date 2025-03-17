#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""
import os
import time

import numpy as np
import tensorflow as tf

from data_process import SquadF1, ModelZooBertDataset, TFSquadV1ModelZooPostTransform, ModelZooBertDataLoader

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    'input_model', None, 'Run inference with specified pb graph.')

flags.DEFINE_string(
    'output_model', None, 'The output model of the quantized model.')

flags.DEFINE_string(
    'mode', 'performance', 'define benchmark mode for accuracy or performance')

flags.DEFINE_bool(
    'tune', False, 'whether to tune the model')

flags.DEFINE_bool(
    'benchmark', False, 'whether to benchmark the model')

flags.DEFINE_bool(
    'strip_iterator', False, 'whether to strip the iterator of the model')

flags.DEFINE_string('dataset_location', None,
                    'location of calibration dataset and evaluate dataset')

flags.DEFINE_integer("batch_size", 64, "run batch size")

flags.DEFINE_integer("iters", 100, "The iteration used for benchmark.")


def evaluate(model, dataloader, metric, postprocess):
    """Custom evaluate function to estimate the accuracy of the bert model.

    Args:
        model (tf.Graph_def): The input model graph

    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    from neural_compressor.tensorflow.quantization.utils.utility import iterator_sess_run
    from neural_compressor.tensorflow.utils import Model, BaseModel
    if not isinstance(model, BaseModel):
        model = Model(model)
    model.input_tensor_names = ['input_ids', 'input_mask', 'segment_ids']
    model.output_tensor_names = ['start_logits', 'end_logits']
    input_tensor = model.input_tensor
    output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                        model.output_tensor[0]
    iteration = -1
    if FLAGS.benchmark and FLAGS.mode == 'performance':
        iteration = FLAGS.iters

    warmup = 5
    latency_list = []
    for idx, (inputs, labels) in enumerate(dataloader):
        # dataloader should keep the order and len of inputs same with input_tensor
        assert len(input_tensor) == len(inputs), \
            'inputs len must equal with input_tensor'
        feed_dict = dict(zip(input_tensor, inputs))
        start_time = time.time()
        predictions = model.sess.run(output_tensor, feed_dict)
        latency_list.append(time.time() - start_time)
        predictions, labels = postprocess((predictions, labels))
        metric.update(predictions, labels)
        if idx + 1 == iteration:
            break

    latency = np.array(latency_list[warmup:]).mean() / FLAGS.batch_size

    if FLAGS.benchmark and FLAGS.mode == 'performance':
        print("Batch size = {}".format(FLAGS.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc

def main(_):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    data_path = os.path.join(FLAGS.dataset_location, 'eval.tf_record')
    label_path = os.path.join(FLAGS.dataset_location, 'dev-v1.1.json')
    vocab_path = os.path.join(FLAGS.dataset_location, 'vocab.txt')

    dataset = ModelZooBertDataset(root=data_path, label_file=label_path)
    dataloader = ModelZooBertDataLoader(dataset=dataset, batch_size=FLAGS.batch_size)
    
    def eval(model):
        metric = SquadF1()
        postprocess = TFSquadV1ModelZooPostTransform(label_file=label_path, vocab_file=vocab_path)
        return evaluate(model, dataloader, metric, postprocess)

    if FLAGS.benchmark:
        if FLAGS.mode == 'performance':
            eval(FLAGS.input_model)
        elif FLAGS.mode == 'accuracy':
            acc_result = eval(FLAGS.input_model)
            print("Batch size = %d" % dataloader.batch_size)
            print("Accuracy: %.5f" % acc_result)

    elif FLAGS.tune:
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model, Model

        model = Model(FLAGS.input_model)
        model.input_tensor_names = ['input_ids', 'input_mask', 'segment_ids']
        model.output_tensor_names = ['start_logits', 'end_logits']
        quant_config = StaticQuantConfig()
        q_model = quantize_model(model, quant_config, dataloader)
        q_model.save(FLAGS.output_model)

if __name__ == "__main__":
    tf.compat.v1.app.run()
