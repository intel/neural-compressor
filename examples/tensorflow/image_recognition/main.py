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
from argparse import ArgumentParser

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

import datasets

# override by args
INPUTS = "input" 
OUTPUTS = "predict" 

INCEPTION_V3_IMAGE_SIZE = 224

tf.compat.v1.disable_eager_execution()

def load_graph(model_file):
  """This is a function to load TF graph from pb file

  Args:
      model_file (string): TF pb file local path

  Returns:
      graph: TF graph object
  """
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  import os
  file_ext = os.path.splitext(model_file)[1]

  with open(model_file, "rb") as f:
    if file_ext == '.pbtxt':
      text_format.Merge(f.read(), graph_def)
    else:
      graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def, name='')

  return graph

class Dataset(object):
  """This is a example class that wrapped the model specified parameters,
    such as dataset path, batch size.
    And more importantly, it provides the ability to iterate the dataset.
  Yields:
      tuple: yield data and label 2 numpy array
  """
  def __init__(self, data_location, subset, input_height, input_width,
                batch_size, num_cores, resize_method='crop', mean_value=[0.0,0.0,0.0], label_adjust=False):
    """dataloader generator

    Args:
        data_location (str): tf recorder local path
        subset (str): training or validation part
        input_height (int): input image size
        input_width (int): input image size
        batch_size (int): dataloader batch size
        num_cores (int): parallel 
        resize_method (str, optional): data preprocession methods. Defaults to 'crop'.
        mean_value (list, optional): data mean value. Defaults to [0.0,0.0,0.0].
        label_adjust (bool, optional): adjust the label value. Defaults to False.
    """
    self.batch_size = batch_size
    self.subset = subset
    self.dataset = datasets.ImagenetData(data_location)
    self.total_image = self.dataset.num_examples_per_epoch(self.subset)
    self.preprocessor = self.dataset.get_image_preprocessor()(
        input_height,
        input_width,
        batch_size,
        num_cores,
        resize_method,
        mean_value)
    self.label_adjust = label_adjust
    self.n = int(self.total_image / self.batch_size)

  def __iter__(self):
    images, labels = self.preprocessor.minibatch(self.dataset, subset=self.subset,
                      cache_data=False)
    # as the batch size if always 1, we will reduce the first dimension
    with tf.compat.v1.Session() as sess:
        for i in range(self.n):
            image, label = sess.run([images, labels])
            if self.label_adjust:
                label -= 1
            image = image.reshape(image.shape[1:])
            label = label.reshape(label.shape[1:])
            yield image, label

class eval_classifier_optimized_graph:
  """Evaluate image classifier with optimized TensorFlow graph"""

  def __init__(self):

    arg_parser = ArgumentParser(description='Parse args')

    arg_parser.add_argument('-b', "--batch-size",
                            help="Specify the batch size. If this " \
                                 "parameter is not specified or is -1, the " \
                                 "largest ideal batch size for the model will " \
                                 "be used.",
                            dest="batch_size", type=int, default=-1)

    arg_parser.add_argument('-e', "--num-inter-threads",
                            help='The number of inter-thread.',
                            dest='num_inter_threads', type=int, default=0)

    arg_parser.add_argument('-a', "--num-intra-threads",
                            help='The number of intra-thread.',
                            dest='num_intra_threads', type=int, default=0)

    arg_parser.add_argument('-g', "--input-graph",
                            help='Specify the input graph for the transform tool',
                            dest='input_graph')
    arg_parser.add_argument("--output-graph",
                            help='Specify tune result model save dir',
                            dest='output_graph')

    arg_parser.add_argument('-i', "--input",
                            help='Specify the input of the model',
                            dest='input')
    arg_parser.add_argument('-o', "--output",
                            help='Specify the output of the model',
                            dest='output')
    arg_parser.add_argument('--image_size', dest='image_size',
                            help='image size',
                            type=int, default=224)
 
    arg_parser.add_argument('-d', "--data-location",
                            help='Specify the location of the data. '
                                 'If this parameter is not specified, '
                                 'the benchmark will use random/dummy data.',
                            dest="data_location", default=None)

    arg_parser.add_argument('-r', "--accuracy-only",
                            help='For accuracy measurement only.',
                            dest='accuracy_only', action='store_true')
    arg_parser.add_argument('--resize_method', help='dataset preprocession',
                            dest='resize_method', default='crop')

    arg_parser.add_argument('--r_mean', help='dataset preprocession',
                            type=float, dest='r_mean', default=0.0)
    arg_parser.add_argument('--g_mean', help='dataset preprocession',
                            type=float, dest='g_mean', default=0.0)
    arg_parser.add_argument('--b_mean', help='dataset preprocession',
                            type=float, dest='b_mean', default=0.0)
    arg_parser.add_argument("--label_adjust", help='Such as RN101 need adjust label',
                            dest='label_adjust', action='store_true')
    arg_parser.add_argument("--warmup-steps", type=int, default=10,
                            help="number of warmup steps")
    arg_parser.add_argument("--steps", type=int, default=50,
                            help="number of steps")
    arg_parser.add_argument("--config", default=None,
                            help="tuning config")
    arg_parser.add_argument(
      '--data-num-inter-threads', dest='data_num_inter_threads',
      help='number threads across operators',
      type=int, default=16)
    arg_parser.add_argument(
      '--data-num-intra-threads', dest='data_num_intra_threads',
      help='number threads for data layer operator',
      type=int, default=14)
    arg_parser.add_argument(
      '--num-cores', dest='num_cores',
      help='number of cores',
      type=int, default=28)
    arg_parser.add_argument(
        '--env', dest='env', help='specific Tensorflow env',
        default='mkl'
    )
    arg_parser.add_argument('--benchmark', dest='benchmark', action='store_true', help='run benchmark')
    arg_parser.add_argument('--tune', dest='tune', action='store_true', help='use ilit to tune.')

    self.args = arg_parser.parse_args()

    # validate the arguments specific for InceptionV3
    self.validate_args()

  def run(self):
      """ This is iLiT function include tuning and benchmark option """

      if self.args.tune:
          q_model = self.auto_tune()
          def save(model, path):
              from tensorflow.python.platform import gfile
              f = gfile.GFile(path, 'wb')
              f.write(model.as_graph_def().SerializeToString())

          save(q_model, evaluate_opt_graph.args.output_graph)

      if self.args.benchmark:
          infer_graph = tf.Graph()
          with infer_graph.as_default():
              graph_def = tf.compat.v1.GraphDef()
              with tf.compat.v1.gfile.FastGFile(self.args.input_graph, 'rb') as input_file:
                  input_graph_content = input_file.read()
                  graph_def.ParseFromString(input_graph_content)

              output_graph = optimize_for_inference(graph_def, [self.args.input],
                                                    [self.args.output], dtypes.float32.as_datatype_enum, False)
              tf.import_graph_def(output_graph, name='')

          self.eval_inference(infer_graph)

  def auto_tune(self):
    """This is iLiT tuning part to generate a quantized pb

    Returns:
        graph: it will return a quantized pb
    """
    import ilit
    fp32_graph = load_graph(self.args.input_graph)
    tuner = ilit.Tuner(self.args.config)

    dataset = Dataset(self.args.data_location, 'validation',
                            self.args.image_size, self.args.image_size,
                            1, self.args.num_cores,
                            self.args.resize_method,  
                            [self.args.r_mean,self.args.g_mean,self.args.b_mean], self.args.label_adjust)
    dataloader = ilit.data.DataLoader('tensorflow', dataset, batch_size=self.args.batch_size)
    q_model = tuner.tune(
                        fp32_graph,
                        q_dataloader=dataloader,
                        # eval_func=self.eval_inference)
                        eval_func=None,
                        eval_dataloader=dataloader)
    return q_model

  def eval_inference(self, infer_graph):
    """run benchmark with optimized graph"""

    print("Run inference")

    data_config = tf.compat.v1.ConfigProto()
    data_config.intra_op_parallelism_threads = self.args.data_num_intra_threads
    data_config.inter_op_parallelism_threads = self.args.data_num_inter_threads
    data_config.use_per_session_threads = 1

    infer_config = tf.compat.v1.ConfigProto()
    if self.args.env == 'mkl':
        print("Set inter and intra for mkl")
        infer_config.intra_op_parallelism_threads = self.args.num_intra_threads
        infer_config.inter_op_parallelism_threads = self.args.num_inter_threads
    infer_config.use_per_session_threads = 1

    data_graph = tf.Graph()
    with data_graph.as_default():
      if (self.args.data_location):
        print("Inference with real data.")
        dataset = datasets.ImagenetData(self.args.data_location)
        preprocessor = dataset.get_image_preprocessor()(
          self.args.image_size, self.args.image_size, self.args.batch_size,
          num_cores=self.args.num_cores,
          resize_method=self.args.resize_method,
          mean_value=[self.args.r_mean,self.args.g_mean,self.args.b_mean])
        images, labels = preprocessor.minibatch(dataset, subset='validation')
      else:
        print("Inference with dummy data.")
        input_shape = [self.args.batch_size, self.args.image_size, self.args.image_size, 3]
        images = tf.random.uniform(input_shape, 0.0, 255.0, dtype=tf.float32, name='synthetic_images')

    # Definite input and output Tensors for detection_graph
    input_tensor = infer_graph.get_tensor_by_name(self.args.input + ':0')
    output_tensor = infer_graph.get_tensor_by_name(self.args.output + ':0')

    data_sess = tf.compat.v1.Session(graph=data_graph,  config=data_config)
    infer_sess = tf.compat.v1.Session(graph=infer_graph, config=infer_config)

    num_processed_images = 0
    num_remaining_images = datasets.IMAGENET_NUM_VAL_IMAGES

    if (not self.args.accuracy_only):
      iteration = 0
      warm_up_iteration = self.args.warmup_steps
      total_run = self.args.steps
      total_time = 0

      while num_remaining_images >= self.args.batch_size and iteration < total_run:
        iteration += 1

        data_load_start = time.time()
        image_np = data_sess.run(images)
        data_load_time = time.time() - data_load_start

        num_processed_images += self.args.batch_size
        num_remaining_images -= self.args.batch_size

        start_time = time.time()
        infer_sess.run([output_tensor], feed_dict={input_tensor: image_np})
        time_consume = time.time() - start_time

        # only add data loading time for real data, not for dummy data
        if self.args.data_location:
          time_consume += data_load_time

        print('Iteration %d: %.6f sec' % (iteration, time_consume))
        if iteration > warm_up_iteration:
          total_time += time_consume

      time_average = total_time / (iteration - warm_up_iteration)
      print('Average time: %.6f sec' % (time_average))

      print('Batch size = %d' % self.args.batch_size)
      if (self.args.batch_size == 1):
        print('Latency: %.3f ms' % (time_average * 1000))

      print('Throughput: %.3f images/sec' % (self.args.batch_size / time_average))

    else:  # accuracy check
      total_accuracy1, total_accuracy5 = (0.0, 0.0)

      while num_remaining_images >= self.args.batch_size:
        # Reads and preprocess data
        np_images, np_labels = data_sess.run([images, labels])
        if self.args.label_adjust:
            np_labels -= 1
        num_processed_images += self.args.batch_size
        num_remaining_images -= self.args.batch_size

        start_time = time.time()
        # Compute inference on the preprocessed data
        predictions = infer_sess.run(output_tensor,
                                     {input_tensor: np_images})
        elapsed_time = time.time() - start_time

        with tf.Graph().as_default() as accu_graph:
          accuracy1 = tf.reduce_sum(
            input_tensor=tf.cast(tf.nn.in_top_k(predictions=tf.constant(predictions),
                                   targets=tf.constant(np_labels), k=1), tf.float32))

          accuracy5 = tf.reduce_sum(
            input_tensor=tf.cast(tf.nn.in_top_k(predictions=tf.constant(predictions),
                                   targets=tf.constant(np_labels), k=5), tf.float32))
          with tf.compat.v1.Session() as accu_sess:
            np_accuracy1, np_accuracy5 = accu_sess.run([accuracy1, accuracy5])

          total_accuracy1 += np_accuracy1
          total_accuracy5 += np_accuracy5

        print("Iteration time: %0.4f ms" % elapsed_time)
        print("Processed %d images. (Top1 accuracy, Top5 accuracy) = (%0.4f, %0.4f)" \
              % (num_processed_images, total_accuracy1 / num_processed_images,
                 total_accuracy5 / num_processed_images))

    print("Accuracy: %.5f" % (total_accuracy1 / num_processed_images))

  
  def validate_args(self):
    """validate the arguments"""

    if not self.args.data_location:
      if self.args.accuracy_only:
        raise ValueError("You must use real data for accuracy measurement.")


if __name__ == "__main__":

  evaluate_opt_graph = eval_classifier_optimized_graph()
  evaluate_opt_graph.run()

  #evaluate_opt_graph.eval_inference(q_model)
