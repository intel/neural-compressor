#!/usr/bin/env python
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
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image pre-processing utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.data.experimental import parallel_interleave
from tensorflow.python.data.experimental import map_and_batch
from tensorflow.python.platform import gfile
import vgg_preprocessing

def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.
  """
  # Dense features in Example proto.
  feature_map = {
    'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                        default_value=''),
    'image/class/label': tf.io.FixedLenFeature([1], dtype=tf.int64,
                                            default_value=-1)
  }
  sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
    {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                 'image/object/bbox/ymin',
                                 'image/object/bbox/xmax',
                                 'image/object/bbox/ymax']})

  features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  return features['image/encoded'], label


def eval_image(image, height, width, resize_method, mean_value,
               central_fraction=0.875, scope=None, scale=1.0):
  with tf.compat.v1.name_scope('eval_image'):
    if resize_method == 'crop':
      shape = tf.shape(input=image)
      image = tf.cond(pred=tf.less(shape[0], shape[1]),
                      true_fn=lambda: tf.image.resize(image,
                                                     tf.convert_to_tensor(value=[256, 256 * shape[1] / shape[0]],
                                                                          dtype=tf.int32)),
                      false_fn=lambda: tf.image.resize(image,
                                                     tf.convert_to_tensor(value=[256 * shape[0] / shape[1], 256],
                                                                          dtype=tf.int32)))
      shape = tf.shape(input=image)
      y0 = (shape[0] - height) // 2
      x0 = (shape[1] - width) // 2
      distorted_image = tf.image.crop_to_bounding_box(image, y0, x0, height, width)
      distorted_image.set_shape([height, width, 3])
      #means = tf.broadcast_to([123.68, 116.78, 103.94], tf.shape(input=distorted_image))
      means = tf.broadcast_to(mean_value, tf.shape(input=distorted_image))
      return (distorted_image - means) * scale
    elif resize_method == 'bilinear':  # bilinear
      if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      # Crop the central region of the image with an area containing 87.5% of
      # the original image.
      if central_fraction:
        image = tf.image.central_crop(image, central_fraction=central_fraction)

      if height and width:
        # Resize the image to the specified height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [height, width],
                                         method=tf.image.ResizeMethod.BILINEAR)
        image = tf.squeeze(image, [0])
      image = tf.subtract(image, 0.5)
      image = tf.multiply(image, 2.0)
      means = tf.broadcast_to(mean_value, tf.shape(input=image))
      return (image - means) * scale
    elif resize_method == 'vgg':
      image = vgg_preprocessing.preprocess_image(image,height,width,False)
      return image * scale
    else:
      raise ValueError('Just support crop, bilinear, vgg for now')

class RecordInputImagePreprocessor(object):
  """Preprocessor for images with RecordInput format."""

  def __init__(self,
               height,
               width,
               batch_size,
               num_cores,
               resize_method="bilinear",
               mean_value=[0.0,0.0,0.0],
               scale=1.0):

    self.height = height
    self.width = width
    self.batch_size = batch_size
    self.num_cores = num_cores
    self.resize_method = resize_method
    self.mean_value = mean_value
    self.scale = scale

  def parse_and_preprocess(self, value):
    # parse
    image_buffer, label_index = parse_example_proto(value)
    # preprocess
    image = tf.image.decode_jpeg(
      image_buffer, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
    image = eval_image(image, self.height, self.width, self.resize_method, self.mean_value, scale=self.scale)

    return (image, label_index)

  def minibatch(self, dataset, subset, cache_data=False):

    with tf.compat.v1.name_scope('batch_processing'):

      glob_pattern = dataset.tf_record_pattern(subset)
      file_names = gfile.Glob(glob_pattern)
      if not file_names:
        raise ValueError('Found no files in --data_dir matching: {}'
                         .format(glob_pattern))
      ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=False)

      ds = ds.apply(
        parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=self.num_cores))

      if cache_data:
        ds = ds.take(1).cache().repeat()

      ds = ds.prefetch(buffer_size=self.batch_size)

      # num of parallel batches not greater than 56
      max_num_parallel_batches = min(56, 2 * self.num_cores)
      ds = ds.apply(
        map_and_batch(
          map_func=self.parse_and_preprocess,
          batch_size=self.batch_size,
          num_parallel_batches=max_num_parallel_batches,
          num_parallel_calls=None))

      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # this number can be tuned

      ds_iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
      images, labels = ds_iterator.get_next()
      # reshape
      labels = tf.reshape(labels, [self.batch_size])

      return images, labels
