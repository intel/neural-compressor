#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import tensorflow as tf
import numpy as np
import os

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
    from neural_compressor.adaptor.tf_utils.util import iterator_sess_run
    from neural_compressor.objective import Performance
    from neural_compressor.model import Model, BaseModel
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
    measurer = Performance()

    warmup = 5
    for idx, (inputs, labels) in enumerate(dataloader):
        # dataloader should keep the order and len of inputs same with input_tensor
        assert len(input_tensor) == len(inputs), \
            'inputs len must equal with input_tensor'
        feed_dict = dict(zip(input_tensor, inputs))
        measurer.start()
        predictions = model.sess.run(output_tensor, feed_dict)
        measurer.end()
        predictions, labels = postprocess((predictions, labels))
        metric.update(predictions, labels)
        if idx + 1 == iteration:
            break

    latency_list = measurer.result_list()
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

    from neural_compressor.metric import SquadF1
    metric = SquadF1()
    from neural_compressor.utils.create_obj_from_config import create_dataloader
    data_path = os.path.join(FLAGS.dataset_location, 'eval.tf_record')
    label_path = os.path.join(FLAGS.dataset_location, 'dev-v1.1.json')
    vocab_path = os.path.join(FLAGS.dataset_location, 'vocab.txt')
    dataloader_args = {
        'batch_size': FLAGS.batch_size,
        'dataset': {"mzbert": {'root': data_path,
                               'label_file': label_path}},
        'transform': None,
        'filter': None
    }
    dataloader = create_dataloader('tensorflow', dataloader_args)
    from neural_compressor.data import TFSquadV1ModelZooPostTransform
    postprocess = TFSquadV1ModelZooPostTransform(label_file=label_path, vocab_file=vocab_path)
    def eval(model):
        return evaluate(model, dataloader, metric, postprocess)
    if FLAGS.benchmark:
        if FLAGS.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(warmup=10, iteration=100, cores_per_instance=4, num_of_instance=1)
            fit(FLAGS.input_model, conf, b_func=eval)
        elif FLAGS.mode == 'accuracy':
            acc_result = eval(FLAGS.input_model)
            print("Batch size = %d" % dataloader.batch_size)
            print("Accuracy: %.5f" % acc_result)

    elif FLAGS.tune:
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig
        conf = PostTrainingQuantConfig(inputs=['input_ids', 'input_mask', 'segment_ids'],
                                       outputs=['start_logits', 'end_logits'],
                                       calibration_sampling_size=[500])
        q_model = quantization.fit(FLAGS.input_model, conf=conf,
                                   calib_dataloader=dataloader, eval_func=eval)
        q_model.save(FLAGS.output_model)

if __name__ == "__main__":
    tf.compat.v1.app.run()
