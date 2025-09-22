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
# tf.config.optimizer.set_experimental_options({'remapping': False})
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

from neural_compressor import Metric
from neural_compressor.data.transforms.transform import ComposeTransform
from neural_compressor.data.datasets.dataset import TensorflowImageRecord
from neural_compressor.data.transforms.imagenet_transform import LabelShift
from neural_compressor.data.dataloaders.tensorflow_dataloader import TensorflowDataLoader
from neural_compressor.data.transforms.imagenet_transform import BilinearImagenetTransform

height = width = 224
eval_dataset = TensorflowImageRecord(root=FLAGS.eval_data, transform=ComposeTransform(transform_list= \
                 [BilinearImagenetTransform(height=height, width=width)]))

eval_dataloader = TensorflowDataLoader(dataset=eval_dataset, batch_size=FLAGS.batch_size)

if FLAGS.calib_data:
    calib_dataset = TensorflowImageRecord(root=FLAGS.calib_data, transform= \
        ComposeTransform(transform_list= [BilinearImagenetTransform(height=height, width=width)]))
    calib_dataloader = TensorflowDataLoader(dataset=calib_dataset, batch_size=10)

def evaluate(model):
    """
    Custom evaluate function to inference the model for specified metric on validation dataset.

    Args:
       model (tf.keras.Model): The input model will be the objection of tf.keras.Model.

    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    infer = model.signatures["serving_default"]
    # print ("infer.inputs: {}".format(infer.inputs))
    output_dict_keys = infer.structured_outputs.keys()
    output_name = list(output_dict_keys )[0]
    postprocess = LabelShift(label_shift=1)
    from neural_compressor import METRICS
    metrics = METRICS('tensorflow')
    metric = metrics['topk']()
    latency_list = []

    def eval_func(dataloader, metric):
        warmup = 5
        iteration = None
        latency_list = []
        if FLAGS.benchmark and FLAGS.mode == 'performance':
            iteration = FLAGS.iters
        predict_fun = tf.function(infer, jit_compile=False)
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = np.array(inputs)
            input_tensor = tf.constant(inputs, dtype=tf.float32)
            input_tensor = tf.transpose(input_tensor, perm=[0, 3, 1, 2])
            start = time.time()
            predictions = predict_fun(input_tensor)[output_name]
            end = time.time()
            predictions, labels = postprocess((predictions, labels))
            predictions = predictions.numpy()
            metric.update(predictions, labels)
            latency_list.append(end - start)
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
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion
        from neural_compressor import set_random_seed
        set_random_seed(9527)
        excluded_op_type = {
                                    'matmul': {
                                        'weight':{
                                            'dtype':['fp32']
                                        },
                                        'activation':{
                                            'dtype':['fp32']
                                        }
                                    }
                                }
        config = PostTrainingQuantConfig(backend='itex',
            calibration_sampling_size=[50, 100],
            accuracy_criterion = AccuracyCriterion(tolerable_loss=0.9999),)
            #op_type_dict=excluded_op_type,)
        q_model = fit(
            model=FLAGS.input_model,
            conf=config,
            calib_func=evaluate,
            eval_func=evaluate)
        q_model.save(FLAGS.output_model)

    if FLAGS.benchmark:
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        if FLAGS.mode == 'performance':
            conf = BenchmarkConfig(backend='itex', cores_per_instance=4, num_of_instance=1)
            fit(FLAGS.input_model, conf, b_func=evaluate)
        else:
            # from neural_compressor.model import Model
            # model = Model(FLAGS.input_model).model
            from tensorflow.python.saved_model import load
            model = load.load(FLAGS.input_model)
            accuracy = evaluate(model)
            logger.info('Batch size = %d' % FLAGS.batch_size)
            logger.info("Accuracy: %.5f" % accuracy)

if __name__ == "__main__":
    tf.compat.v1.app.run()
