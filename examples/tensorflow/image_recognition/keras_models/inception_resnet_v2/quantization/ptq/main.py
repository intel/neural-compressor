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
import shutil
import numpy as np
from argparse import ArgumentParser
from neural_compressor import data
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

flags.DEFINE_string(
    'config', 'bert.yaml', 'yaml configuration of the model')

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
  calib_dataset = TensorflowImageRecord(root=FLAGS.calib_data, transform=ComposeTransform(transform_list= \
    [BilinearImagenetTransform(height=299, width=299)]))
  calib_dataloader = DefaultDataLoader(dataset=calib_dataset, batch_size=10)

def evaluate(model, measurer=None):
  """
  Custom Evaluate function to inference the model for specified metric on validation dataset.

  Args:
      model ([tf.saved_model.load]): The model will be the class of tf.saved_model.load(quantized_model_path).
      measurer (object, optional): for precise benchmark measurement.

  Returns:
      [float]: evaluation result, the larger is better.
  """
  infer = model.signatures["serving_default"]
  output_dict_keys = infer.structured_outputs.keys()
  output_name = list(output_dict_keys )[0]
  postprocess = LabelShift(label_shift=1)
  metric = TensorflowTopK(k=1)
  
  def eval_func(dataloader, metric):
      results = []
      for idx, (inputs, labels) in enumerate(dataloader):
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

  results = eval_func(eval_dataloader, metric)
  acc = metric.result()
  return acc

def main(_):
  if FLAGS.tune:
    from neural_compressor.experimental import Quantization, common
    quantizer = Quantization(FLAGS.config)
    quantizer.model = common.Model(FLAGS.input_model)
    quantizer.eval_func = evaluate
    quantizer.calib_dataloader = calib_dataloader
    q_model = quantizer.fit()
    q_model.save(FLAGS.output_model)


  if FLAGS.benchmark:
    from neural_compressor.experimental import Benchmark, common
    evaluator = Benchmark(FLAGS.config)
    evaluator.model = common.Model(FLAGS.input_model)
    evaluator.b_func = evaluate
    evaluator.b_dataloader = eval_dataloader
    evaluator(FLAGS.mode)

if __name__ == "__main__":
    tf.compat.v1.app.run()
