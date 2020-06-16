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
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from random import randint

from tensorflow.python.ops import data_flow_ops
import cnn_util


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
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
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
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox, features['image/class/text']


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
    with tf.name_scope(scope or 'decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3,
                                     fancy_upscaling=False,
                                     dct_method='INTEGER_FAST')

        # image = tf.Print(image, [tf.shape(image)], 'Image shape: ')

        return image


def eval_image(image, height, width, bbox, thread_id, resize):
    """Get the image for model evaluation."""
    with tf.name_scope('eval_image'):
        if not thread_id:
            tf.summary.image(
                'original_image', tf.expand_dims(image, 0))

        if resize == 'crop':
            # Note: This is much slower than crop_to_bounding_box
            #         It seems that the redundant pad step has huge overhead
            # distorted_image = tf.image.resize_image_with_crop_or_pad(image,
            #                                                         height, width)
            shape = tf.shape(image)
            image = tf.cond(tf.less(shape[0], shape[1]),
                            lambda: tf.image.resize_images(image, tf.convert_to_tensor([256, 256 * shape[1] / shape[0]],
                                                                                       dtype=tf.int32)),
                            lambda: tf.image.resize_images(image, tf.convert_to_tensor([256 * shape[0] / shape[1], 256],
                                                                                       dtype=tf.int32)))
            shape = tf.shape(image)

            y0 = (shape[0] - height) // 2
            x0 = (shape[1] - width) // 2
            # y0=tf.random_uniform([],minval=0,maxval=(shape[0] - height + 1), dtype=tf.int32)
            # x0=tf.random_uniform([],minval=0,maxval=(shape[1] - width + 1), dtype=tf.int32)
            ## distorted_image = tf.slice(image, [y0,x0,0], [height,width,3])
            distorted_image = tf.image.crop_to_bounding_box(image, y0, x0, height,
                                                            width)
        else:
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=bbox,
                min_object_covered=0.5,
                aspect_ratio_range=[0.90, 1.10],
                area_range=[0.10, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, _ = sample_distorted_bounding_box
            # Crop the image to the specified bounding box.
            distorted_image = tf.slice(image, bbox_begin, bbox_size)
            resize_method = {
                'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                'bilinear': tf.image.ResizeMethod.BILINEAR,
                'bicubic': tf.image.ResizeMethod.BICUBIC,
                'area': tf.image.ResizeMethod.AREA
            }[resize]
            # This resizing operation may distort the images because the aspect
            # ratio is not respected.
            if cnn_util.tensorflow_version() >= 11:
                distorted_image = tf.image.resize_images(
                    distorted_image, [height, width],
                    resize_method,
                    align_corners=False)
            else:
                distorted_image = tf.image.resize_images(
                    distorted_image, height, width, resize_method, align_corners=False)
        distorted_image.set_shape([height, width, 3])
        if not thread_id:
            tf.summary.image(
                'cropped_resized_image', tf.expand_dims(distorted_image, 0))
        image = distorted_image
    return image


def distort_image(image, height, width, bbox, thread_id=0, scope=None):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
      image: 3-D float Tensor of image
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      thread_id: integer indicating the preprocessing thread.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    # with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
    # with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
    with tf.name_scope(scope or 'distort_image'):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Display the bounding box in the first thread only.
        if not thread_id:
            image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                          bbox)
            tf.summary.image(
                'image_with_bounding_boxes', image_with_box)

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an allowed
        # range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.99, 1.01],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        if not thread_id:
            image_with_distorted_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), distort_bbox)
            tf.summary.image(
                'images_with_distorted_bounding_box',
                image_with_distorted_box)

        # Crop the image to the specified bounding box.
        distorted_image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        resize_method = thread_id % 4
        if cnn_util.tensorflow_version() >= 11:
            distorted_image = tf.image.resize_images(
                distorted_image, [height, width], resize_method, align_corners=False)
        else:
            distorted_image = tf.image.resize_images(
                distorted_image, height, width, resize_method, align_corners=False)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])
        if not thread_id:
            tf.summary.image(
                'cropped_resized_image',
                tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors.
        distorted_image = distort_color(distorted_image, thread_id)

        # Note: This ensures the scaling matches the output of eval_image
        distorted_image *= 256

        if not thread_id:
            tf.summary.image(
                'final_distorted_image',
                tf.expand_dims(distorted_image, 0))
        return distorted_image


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    # with tf.op_scope([image], scope, 'distort_color'):
    # with tf.name_scope(scope, 'distort_color', [image]):
    with tf.name_scope(scope or 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


class ImagePreprocessor(object):
    """Preprocessor for input images."""

    def __init__(self,
                 height,
                 width,
                 batch_size,
                 device_count,
                 dtype=tf.float32,
                 train=True,
                 distortions=None,
                 resize_method=None):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.device_count = device_count
        self.dtype = dtype
        self.train = train
        self.resize_method = resize_method
        if distortions is None:
            distortions = False
        self.distortions = distortions
        if self.batch_size % self.device_count != 0:
            raise ValueError(
                ('batch_size must be a multiple of device_count: '
                 'batch_size %d, device_count: %d') %
                (self.batch_size, self.device_count))
        self.batch_size_per_device = self.batch_size // self.device_count

    def preprocess(self, image_buffer, bbox, thread_id):
        """Preprocessing image_buffer using thread_id."""
        # Note: Width and height of image is known only at runtime.
        image = tf.image.decode_jpeg(image_buffer, channels=3,
                                     dct_method='INTEGER_FAST')
        if self.train and self.distortions:
            image = distort_image(image, self.height, self.width, bbox, thread_id)
        else:
            image = eval_image(image, self.height, self.width, bbox, thread_id,
                               self.resize_method)
        # Note: image is now float32 [height,width,3] with range [0, 255]

        # image = tf.cast(image, tf.uint8) # HACK TESTING

        return image

    def minibatch(self, dataset, subset):
        with tf.name_scope('batch_processing'):
            images = [[] for i in range(self.device_count)]
            labels = [[] for i in range(self.device_count)]
            record_input = data_flow_ops.RecordInput(
                file_pattern=dataset.tf_record_pattern(subset),
                seed=randint(0, 9000),
                parallelism=64,
                buffer_size=10000,
                batch_size=self.batch_size,
                name='record_input')
            records = record_input.get_yield_op()
            records = tf.split(records, self.batch_size, 0)
            records = [tf.reshape(record, []) for record in records]
            for i in xrange(self.batch_size):
                value = records[i]
                image_buffer, label_index, bbox, _ = parse_example_proto(value)
                image = self.preprocess(image_buffer, bbox, i % 4)
                device_index = i % self.device_count
                images[device_index].append(image)
                labels[device_index].append(label_index)
            label_index_batch = [None] * self.device_count
            for device_index in xrange(self.device_count):
                images[device_index] = tf.parallel_stack(images[device_index])
                label_index_batch[device_index] = tf.concat(labels[device_index], 0)

                # dynamic_pad=True) # HACK TESTING dynamic_pad=True
                images[device_index] = tf.cast(images[device_index], self.dtype)
                depth = 3
                images[device_index] = tf.reshape(
                    images[device_index],
                    shape=[self.batch_size_per_device, self.height, self.width, depth])
                label_index_batch[device_index] = tf.reshape(
                    label_index_batch[device_index], [self.batch_size_per_device])
                # Display the training images in the visualizer.
                # tf.summary.image('images', images)

            return images, label_index_batch
