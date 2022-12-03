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

from neural_compressor.experimental.metric.metric import TensorflowTopK
from neural_compressor.experimental.data.transforms.transform import ComposeTransform
from neural_compressor.experimental.data.datasets.dataset import TensorflowImageRecord
from neural_compressor.experimental.data.transforms.imagenet_transform import LabelShift
from neural_compressor.experimental.data.dataloaders.default_dataloader import DefaultDataLoader
from neural_compressor.data.transforms.imagenet_transform import BilinearImagenetTransform

eval_dataset = TensorflowImageRecord(root=FLAGS.eval_data, transform=ComposeTransform(transform_list= \
                [BilinearImagenetTransform(height=299, width=299)]))
if FLAGS.benchmark and FLAGS.mode == 'performance':
    eval_dataloader = DefaultDataLoader(dataset=eval_dataset, batch_size=1)
else:
    eval_dataloader = DefaultDataLoader(dataset=eval_dataset, batch_size=32)
if FLAGS.calib_data:
    calib_dataset = TensorflowImageRecord(root=FLAGS.calib_data, transform= \
        ComposeTransform(transform_list= [BilinearImagenetTransform(height=299, width=299)]))
    calib_dataloader = DefaultDataLoader(dataset=calib_dataset, batch_size=10)

def evaluate(model, measurer=None):
    """
    Custom evaluate function to inference the model for specified metric on validation dataset.

    Args:
        model (tf.saved_model.load): The input model will be the class of tf.saved_model.load(quantized_model_path).
        measurer (object, optional): for benchmark measurement of duration.

    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    infer = model.signatures["serving_default"]
    output_dict_keys = infer.structured_outputs.keys()
    output_name = list(output_dict_keys )[0]
    postprocess = LabelShift(label_shift=1)
    metric = TensorflowTopK(k=1)

    def eval_func(dataloader, metric):
        results = []
        for _, (inputs, labels) in enumerate(dataloader):
            inputs = np.array(inputs)
            input_tensor = tf.constant(inputs)
            if measurer:
                measurer.start()
            predictions = infer(input_tensor)[output_name]
            if measurer:
                measurer.end()
            predictions = predictions.numpy()
            predictions, labels = postprocess((predictions, labels))
            metric.update(predictions, labels)
        return results

    _ = eval_func(eval_dataloader, metric)
    acc = metric.result()
    return acc

def main(_):
    if FLAGS.tune:
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig, \
            TuningCriterion, AccuracyCriterion, AccuracyLoss, set_random_seed
        set_random_seed(9527)
        tuning_criterion = TuningCriterion(
            strategy="basic",
            timeout=0,
            max_trials=100,
            objective="performance")
        tolerable_loss = AccuracyLoss(loss=0.01)
        accuracy_criterion = AccuracyCriterion(
            higher_is_better=True,
            criterion='relative',
            tolerable_loss=tolerable_loss)
        config = PostTrainingQuantConfig(
            device="cpu",
            backend="tensorflow",
            inputs=[],
            outputs=[],
            approach="static",
            calibration_sampling_size=[50, 100],
            op_type_list=None,
            op_name_list=None,
            reduce_range=None,
            extra_precisions=[],
            tuning_criterion=tuning_criterion,
            accuracy_criterion=accuracy_criterion)
        q_model = fit(
            model=FLAGS.input_model,
            conf=config,
            calib_dataloader=calib_dataloader,
            calib_func=None,
            eval_dataloader=eval_dataloader,
            eval_func=evaluate,
            eval_metric=None)
        q_model.save(FLAGS.output_model)

    if FLAGS.benchmark:
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        conf = BenchmarkConfig(warmup=5, iteration=100, cores_per_instance=4, num_of_instance=7)
        fit(FLAGS.input_model, conf, b_dataloader=eval_dataloader, b_func=evaluate)

if __name__ == "__main__":
    tf.compat.v1.app.run()
