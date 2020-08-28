#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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
#

import os
import io
import skimage.io
import glob
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
import time
import tf_slim as slim

flags = tf.flags
flags.DEFINE_string('style_images_paths', None, 'Paths to the style images'
                    'for evaluation.')
flags.DEFINE_string('content_images_paths', None, 'Paths to the content images'
                    'for evaluation.')
flags.DEFINE_string('output_dir', './result', 'Output stylized image directory.')

flags.DEFINE_string('output_model', None, 'Output model directory.')

flags.DEFINE_string('input_model', None, 'Output directory.')

flags.DEFINE_string('precision', 'fp32', 'precision')

flags.DEFINE_string('config', None, 'yaml configuration for tuning')

FLAGS = flags.FLAGS

def load_img(path, resize_shape=(256, 256), crop_ratio=0.1):
    img = Image.open(path)
    width, height = img.size
    crop_box = (crop_ratio*height, crop_ratio*width, (1-crop_ratio)*height, (1-crop_ratio)*width)
    img = np.asarray(img.crop(crop_box).resize(resize_shape))
    if img.max() > 1.0:
        img = img / 255.
    img = img.astype(np.float32)[np.newaxis, ...]
    return img

def save_image(image, output_file, save_format='jpeg'):
    image = np.uint8(image * 255.0)
    buf = io.BytesIO()
    skimage.io.imsave(buf, np.squeeze(image, 0), format=save_format)
    buf.seek(0)
    f = tf.gfile.GFile(output_file, 'w')
    f.write(buf.getvalue())
    f.close()

def image_style_transfer(sess, content_img_path, style_img_path):
    stylized_images = sess.graph.get_tensor_by_name('import/import/transformer/expand/conv3/conv/Sigmoid:0')
    style_img_np = load_img(style_img_path, crop_ratio=0)
    content_img_np = load_img(content_img_path, crop_ratio=0)
    stylized_image_res = sess.run(
        stylized_images,
        feed_dict={
            'import/import/style_input:0': style_img_np,
            'import/import/content_input:0': content_img_np})
    # saves stylized image.
    save_image(stylized_image_res, os.path.join(FLAGS.output_dir, 'stylized_image.jpg'))

def _parse_ckpt_bn_input(graph_def):
    """parse ckpt batch norm inputs to match correct moving mean and variance
    Args:
        graph_def (graph_def): original graph_def
    Returns:
        graph_def: well linked graph_def 
    """
    for node in graph_def.node:
        if node.op == 'FusedBatchNorm':
            moving_mean_op_name = node.input[3]
            moving_var_op_name = node.input[4]
            moving_mean_op = _get_nodes_from_name(moving_mean_op_name, graph_def)[0]
            moving_var_op = _get_nodes_from_name(moving_var_op_name, graph_def)[0]

            if moving_mean_op.op == 'Const':
                name_part = moving_mean_op_name.rsplit('/', 1)[0]
                real_moving_mean_op_name = name_part + '/moving_mean'
                if len(_get_nodes_from_name(real_moving_mean_op_name, graph_def)) > 0:
                    # replace the real moving mean op name
                    node.input[3] = real_moving_mean_op_name

            if moving_var_op.op == 'Const':
                name_part = moving_var_op_name.rsplit('/', 1)[0]
                real_moving_var_op_name = name_part + '/moving_variance'
                if len(_get_nodes_from_name(real_moving_var_op_name, graph_def)) > 0:
                    # replace the real moving mean op name
                    node.input[4] = real_moving_var_op_name

    return graph_def

def _get_nodes_from_name(node_name, graph_def):
    return [node for node in graph_def.node if node.name==node_name]
 
def main(args=None):
  tf.logging.set_verbosity(tf.logging.INFO)
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MkDir(FLAGS.output_dir)

  with tf.Session() as sess:
      if FLAGS.input_model.rsplit('.', 1)[-1] == 'ckpt':
          style_img_ph = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='style_input')
          content_img_ph = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='content_input')
          # import meta_graph
          meta_data_path = FLAGS.input_model + '.meta'
          saver = tf.train.import_meta_graph(meta_data_path, clear_devices=True)

          sess.run(tf.global_variables_initializer())
          saver.restore(sess, FLAGS.input_model)
          graph_def = sess.graph.as_graph_def()
          _parse_ckpt_bn_input(graph_def)

          replace_style = 'style_image_processing/ResizeBilinear_2'
          replace_content = 'batch_processing/batch'
          for node in graph_def.node:
              for idx, input_name in enumerate(node.input):
                  # replace style input and content input nodes to  placeholder
                  if replace_content == input_name:
                      node.input[idx] = 'content_input'
                  if replace_style == input_name:
                      node.input[idx] = 'style_input'

          output_name = 'transformer/expand/conv3/conv/Sigmoid'
          frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, [output_name])
      # use frozen pb instead 
      elif FLAGS.input_model.rsplit('.', 1)[-1] == 'pb':
          with open(FLAGS.input_model, 'rb') as f:
              frozen_graph = tf.GraphDef()
              frozen_graph.ParseFromString(f.read())
      else:
          print("not supported model format")
          exit(-1)

      with tf.Graph().as_default() as graph:
          tf.import_graph_def(frozen_graph)
          import ilit
          tuner = ilit.Tuner(FLAGS.config)
          quantized_model = tuner.tune(graph, eval_func=eval_func)

          # save the frozen model for deployment
          with tf.io.gfile.GFile(os.path.join(FLAGS.output_model, "int8.pb"), "wb") as f:
              f.write(quantized_model.as_graph_def().SerializeToString())

  # validate the quantized model here
  with tf.Graph().as_default(), tf.Session() as sess:
      # create dataloader using default style_transfer dataset and generate stylized images
      dataset = ilit.data.DATASETS('tensorflow')['style_transfer'](FLAGS.content_images_paths,
                                                                   FLAGS.style_images_paths,
                                                                   crop_ratio=0.2,
                                                                   resize_shape=(256, 256))
      dataloader = ilit.data.DataLoader('tensorflow', dataset=dataset)
      # # test your fp32 graph and compare with quantized one
      # tf.import_graph_def(frozen_graph)
      # style_transfer(sess, dataloader, 'fp32')

      tf.import_graph_def(quantized_model.as_graph_def())
      style_transfer(sess, dataloader, 'quantized')

def add_import_to_name(sess, name, try_cnt=2):
    for i in range(0, try_cnt):
        try:
            sess.graph.get_tensor_by_name(name)   
            return name
        except:
            name = 'import/' + name

    raise ValueError('can not find tensor by name')

# validate and  save the files
def style_transfer(sess, dataloader, precision='fp32'):
      time_list = []
      output_name = add_import_to_name(sess, 'transformer/expand/conv3/conv/Sigmoid:0', 3)
      style_name = add_import_to_name(sess, 'style_input:0', 3)
      content_name = add_import_to_name(sess, 'content_input:0', 3)

      stylized_images = sess.graph.get_tensor_by_name(output_name)
      
      for (content_img_np, style_img_np), _ in dataloader:
          start_time = time.time()
          stylized_image_res = sess.run(
              stylized_images,
              feed_dict={
                  style_name: style_img_np,
                  content_name: content_img_np})
          duration = time.time() - start_time
          time_list.append(duration)
          # Saves stylized image.
          save_image(stylized_image_res, os.path.join(FLAGS.output_dir,
                                                      '{}_{}.jpg'.format(precision, len(time_list))))
      warm_up = 2
      throughput = (len(time_list) - warm_up)/ np.array(time_list[warm_up:]).sum()
      print("{} realtime throughput is {}".format(precision, throughput))

def eval_func(model):
    return 1.

def run_tuning():
  tf.disable_v2_behavior()
  tf.app.run(main)

if __name__ == '__main__':
  run_tuning()
