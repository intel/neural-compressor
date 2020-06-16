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
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
import cnn_util

from tensorflow.python.ops import control_flow_ops


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
  
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:
  
      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>
  
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
  
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.io.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
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

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

    return features['image/encoded'], label, bbox, features['image/class/text']


def get_image_resize_method(resize_method, batch_position=0):
    """Get tensorflow resize method.
  
    If resize_method is 'round_robin', return different methods based on batch
    position in a round-robin fashion. NOTE: If the batch size is not a multiple
    of the number of methods, then the distribution of methods will not be
    uniform.
  
    Args:
      resize_method: (string) nearest, bilinear, bicubic, area, or round_robin.
      batch_position: position of the image in a batch. NOTE: this argument can
        be an integer or a tensor
    Returns:
      one of resize type defined in tf.image.ResizeMethod.
    """
    resize_methods_map = {
        'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'bicubic': tf.image.ResizeMethod.BICUBIC,
        'area': tf.image.ResizeMethod.AREA
    }

    if resize_method != 'round_robin':
        return resize_methods_map[resize_method]

    # return a resize method based on batch position in a round-robin fashion.
    resize_methods = resize_methods_map.values()

    def lookup(index):
        return resize_methods[index]

    def resize_method_0():
        return utils.smart_cond(batch_position % len(resize_methods) == 0,
                                lambda: lookup(0), resize_method_1)

    def resize_method_1():
        return utils.smart_cond(batch_position % len(resize_methods) == 1,
                                lambda: lookup(1), resize_method_2)

    def resize_method_2():
        return utils.smart_cond(batch_position % len(resize_methods) == 2,
                                lambda: lookup(2), lambda: lookup(3))

    # NOTE(jsimsa): Unfortunately, we cannot use a single recursive function here
    # because TF would not be able to construct a finite graph.

    return resize_method_0()


def decode_jpeg(image_buffer, scope=None):  # , dtype=tf.float32):
    """Decode a JPEG string into one 3-D float image Tensor.
  
    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    # with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
    # with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
    with tf.compat.v1.name_scope(scope or 'decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)  # ,
        #     fancy_upscaling=False,
        #     dct_method='INTEGER_FAST')

        # image = tf.Print(image, [tf.shape(image)], 'Image shape: ')
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        return image


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
    """Prepare one image for evaluation.
  
    If height and width are specified it would output an image with that size by
    applying resize_bilinear.
  
    If central_fraction is specified it would crop the central fraction of the
    input image.
  
    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      central_fraction: Optional Float, fraction of the image to crop.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.compat.v1.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image,
                                          central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.compat.v1.image.resize_bilinear(image, [height, width],
                                                       align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  
    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.
  
    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.
  
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
  
    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.compat.v1.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
  
    See `tf.image.sample_distorted_bounding_box` for more documentation.
  
    Args:
      image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
        image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding box
        supplied.
      aspect_ratio_range: An optional list of `floats`. The cropped area of the
        image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `floats`. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
      scope: Optional scope for name_scope.
    Returns:
      A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.compat.v1.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            image_size=tf.shape(input=image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox


def preprocess_for_train(image, height, width, bbox,
                         batch_position,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):
    """Distort one image for training a network.
  
    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.
  
    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      batch_position: position of the image in a batch, which affects how images
        are distorted and resized. NOTE: this argument can be an integer or a
        tensor
      scope: Optional scope for op_scope.
      add_image_summaries: Enable image summaries.
    Returns:
      3-D float Tensor of distorted image used for training with range [-1, 1].
    """

    with tf.compat.v1.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                               dtype=tf.float32,
                               shape=[1, 1, 4])
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                      bbox)
        if add_image_summaries:
            tf.compat.v1.summary.image('image_with_bounding_boxes', image_with_box)

        distorted_image, distorted_bbox = distorted_bounding_box_crop(image,
                                                                      bbox)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 3])
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distorted_bbox)
        if add_image_summaries:
            tf.compat.v1.summary.image('images_with_distorted_bounding_box',
                             image_with_distorted_box)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.

        # We select only 1 case for fast_mode bilinear.
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize(x, [height, width],
                                                     method),
            num_cases=num_resize_cases)

        if add_image_summaries:
            tf.compat.v1.summary.image('cropped_resized_image',
                             tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # Randomly distort the colors. There are 1 or 4 ways to do it.
        num_distort_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, ordering: distort_color(x, ordering, fast_mode),
            num_cases=num_distort_cases)

        if add_image_summaries:
            tf.compat.v1.summary.image('final_distorted_image',
                             tf.expand_dims(distorted_image, 0))
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)
        return distorted_image


def distort_color(image, batch_position=0, distort_color_in_yiq=False,
                  scope=None):
    """Distort the color of the image.
  
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops based on the position of the image in a batch.
  
    Args:
      image: float32 Tensor containing single image. Tensor values should be in
        range [0, 1].
      batch_position: the position of the image in a batch. NOTE: this argument
        can be an integer or a tensor
      distort_color_in_yiq: distort color of input images in YIQ space.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    with tf.compat.v1.name_scope(scope or 'distort_color'):
        def distort_fn_0(image=image):
            """Variant 0 of distort function."""
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            # if distort_color_in_yiq:
            #  image = distort_image_ops.random_hsv_in_yiq(
            #      image, lower_saturation=0.5, upper_saturation=1.5,
            #      max_delta_hue=0.2 * math.pi)
            # else:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            return image

        def distort_fn_1(image=image):
            """Variant 1 of distort function."""
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            # if distort_color_in_yiq:
            #  image = distort_image_ops.random_hsv_in_yiq(
            #      image, lower_saturation=0.5, upper_saturation=1.5,
            #      max_delta_hue=0.2 * math.pi)
            # else:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            return image

        image = utils.smart_cond(batch_position % 2 == 0, distort_fn_0,
                                 distort_fn_1)
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


class RecordInputImagePreprocessor(object):
    """Preprocessor for images with RecordInput format."""

    def __init__(self,
                 height,
                 width,
                 batch_size,
                 num_splits,
                 dtype,
                 train,
                 distortions=False,
                 resize_method="bilinear",
                 shift_ratio=0,
                 summary_verbosity=1,
                 distort_color_in_yiq=False,
                 fuse_decode_and_crop=False):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.num_splits = num_splits
        self.dtype = dtype
        self.train = train
        self.resize_method = resize_method
        self.shift_ratio = shift_ratio
        self.distortions = distortions
        self.distort_color_in_yiq = distort_color_in_yiq
        self.fuse_decode_and_crop = fuse_decode_and_crop
        if self.batch_size % self.num_splits != 0:
            raise ValueError(
                ('batch_size must be a multiple of num_splits: '
                 'batch_size %d, num_splits: %d') %
                (self.batch_size, self.num_splits))
        self.batch_size_per_split = self.batch_size // self.num_splits
        self.summary_verbosity = summary_verbosity

    def center_crop(self, img, init_h, init_w):
        height, width, _ = img.shape

        left = int((width - init_w) // 2)
        right = int((width + init_w) // 2)
        top = int((height - init_h) // 2)
        bottom = int((height + init_h) // 2)

        img = img[top: bottom, left: right]

        return img

    def image_preprocess(self, image_buffer, bbox, batch_position):
        """Preprocessing image_buffer as a function of its batch position."""
        if self.train:
            image_buffer = tf.image.decode_jpeg(
                image_buffer, channels=3, dct_method='INTEGER_FAST')
            image = preprocess_for_train(image_buffer, self.height, self.width,
                                         bbox,
                                         batch_position)
        else:
            image = tf.image.decode_jpeg(
                image_buffer, channels=3, dct_method='INTEGER_FAST')

            new_height = int(100. * self.height / 87.5)
            new_width = int(100. * self.width / 87.5)

            if(self.height > self.width):
                w = new_width
                h = int(new_height * self.height / self.width)
            else:
                h = new_height
                w = int(new_width * self.width / self.height)

            image = preprocess_for_eval(image, h, w)
            image = self.center_crop(image, self.height, self.width)

        return image

    def parse_and_preprocess(self, value, batch_position):
        image_buffer, label_index, bbox, _ = parse_example_proto(value)
        image = self.image_preprocess(image_buffer, bbox, batch_position)
        return (label_index, image)

    def minibatch(self, dataset, subset, use_datasets, cache_data,
                  shift_ratio=-1):
        if shift_ratio < 0:
            shift_ratio = self.shift_ratio
        with tf.compat.v1.name_scope('batch_processing'):
            # Build final results per split.
            images = [[] for _ in range(self.num_splits)]
            labels = [[] for _ in range(self.num_splits)]
            if use_datasets:
                glob_pattern = dataset.tf_record_pattern(subset)
                file_names = gfile.Glob(glob_pattern)
                if not file_names:
                    raise ValueError(
                        'Found no files in --data_dir matching: {}'
                        .format(glob_pattern))
                ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=False)
                ds = ds.apply(
                    interleave_ops.parallel_interleave(
                        tf.data.TFRecordDataset, cycle_length=10))
                if cache_data:
                    ds = ds.take(1).cache().repeat()
                counter = tf.data.Dataset.range(self.batch_size)
                counter = counter.repeat()
                ds = tf.data.Dataset.zip((ds, counter))
                ds = ds.prefetch(buffer_size=self.batch_size)
                ds = ds.repeat()
                ds = ds.apply(
                    batching.map_and_batch(
                        map_func=self.parse_and_preprocess,
                        batch_size=self.batch_size_per_split,
                        num_parallel_batches=self.num_splits))
                ds = ds.prefetch(buffer_size=self.num_splits)
                ds_iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
                for d in xrange(self.num_splits):
                    labels[d], images[d] = ds_iterator.get_next()

            else:
                record_input = data_flow_ops.RecordInput(
                    file_pattern=dataset.tf_record_pattern(subset),
                    seed=301,
                    parallelism=64,
                    buffer_size=10000,
                    batch_size=self.batch_size,
                    shift_ratio=shift_ratio,
                    name='record_input')
                records = record_input.get_yield_op()
                records = tf.split(records, self.batch_size, 0)
                records = [tf.reshape(record, []) for record in records]
                for idx in xrange(self.batch_size):
                    value = records[idx]
                    (label, image) = self.parse_and_preprocess(value, idx)
                    split_index = idx % self.num_splits
                    labels[split_index].append(label)
                    images[split_index].append(image)

            for split_index in xrange(self.num_splits):
                if not use_datasets:
                    images[split_index] = tf.parallel_stack(
                        images[split_index])
                    labels[split_index] = tf.concat(labels[split_index], 0)
                images[split_index] = tf.cast(images[split_index], self.dtype)
                depth = 3
                images[split_index] = tf.reshape(
                    images[split_index],
                    shape=[self.batch_size_per_split, self.height, self.width,
                           depth])
                labels[split_index] = tf.reshape(labels[split_index],
                                                 [self.batch_size_per_split])
            return images, labels
