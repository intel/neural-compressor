#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

import numpy as np
from lpot.utils import logger
from lpot.utils.utility import LazyImport
from lpot.experimental.data.transforms import transform_registry, BaseTransform

tf = LazyImport('tensorflow')
cv2 = LazyImport('cv2')

# BELOW IS TO BE DEPRECATED!
@transform_registry(transform_type="ParseDecodeImagenet", \
                    process="preprocess", framework="tensorflow")
class ParseDecodeImagenetTransform(BaseTransform):

    def __call__(self, sample):
        logger.warning('This transform is going to be deprecated, '
            'imagenet decoding will be performed automatically' )
        return sample

@transform_registry(transform_type="ResizeCropImagenet", \
                    process="preprocess", framework="tensorflow")
class TensorflowResizeCropImagenetTransform(BaseTransform):
    """Combination of a series of transforms which is applicable to images in Imagenet.

    Args:
        height (int): Height of the result
        width (int): Width of the result
        random_crop (bool, default=False): whether to random crop
        resize_side (int, default=256):desired shape after resize operation
        random_flip_left_right (bool, default=False): whether to random flip left and right
        mean_value (list, default=[0.0,0.0,0.0]):means for each channel
        scale (float, default=1.0):std value

    Returns:
        tuple of processed image and label
    """

    def __init__(self, height, width, random_crop=False, resize_side=256, \
                 random_flip_left_right=False, mean_value=[0.0,0.0,0.0], scale=1.0):

        self.height = height
        self.width = width
        self.mean_value = mean_value
        self.scale = scale
        self.random_crop = random_crop
        self.random_flip_left_right = random_flip_left_right
        self.resize_side = resize_side

    # sample is (images, labels)
    def __call__(self, sample):
        image, label = sample
        shape = tf.shape(input=image)
        height = tf.cast(shape[0], dtype=tf.float32)
        width = tf.cast(shape[1], dtype=tf.float32)
        scale = tf.cond(pred=tf.greater(height, width), \
                        true_fn=lambda: self.resize_side / width,
                        false_fn=lambda: self.resize_side / height,)

        scale = tf.cast(scale, dtype=tf.float32)
        new_height = tf.cast(tf.math.rint(height*scale), dtype=tf.int32)
        new_width = tf.cast(tf.math.rint(width*scale), dtype=tf.int32)

        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [new_height, new_width],
                                method=tf.image.ResizeMethod.BILINEAR)
        image = tf.squeeze(image)

        # image = tf.cond(pred=tf.greater(shape[0], shape[1]), \
        #                 false_fn=lambda: tf.image.resize(image, \
        #                     tf.convert_to_tensor(value=[self.resize_side*shape[0]/shape[1], \
        #                         self.resize_side], dtype=tf.int32)),
        #                 true_fn=lambda: tf.image.resize(image, \
        #                     tf.convert_to_tensor(value=[self.resize_side, \
        #                         self.resize_side * shape[1] / shape[0]], dtype=tf.int32)),
        #                )

        shape = tf.shape(input=image)
        if self.random_crop:
            y0 = tf.random.uniform(shape=[], minval=0, maxval=(shape[0] - self.height +1), 
                    dtype=tf.dtypes.int32)
            x0 = tf.random.uniform(shape=[], minval=0, maxval=(shape[1] - self.width +1), 
                    dtype=tf.dtypes.int32)
        else:
            y0 = (shape[0] - self.height) // 2
            x0 = (shape[1] - self.width) // 2

        image = tf.image.crop_to_bounding_box(image, y0, x0, self.height, self.width)
        image.set_shape([self.height, self.width, 3])
        if self.random_flip_left_right:
            image = tf.image.random_flip_left_right(image)
        means = tf.broadcast_to(self.mean_value, tf.shape(input=image))
        image = (image - means) * self.scale
        return (image, label)

@transform_registry(transform_type="BilinearImagenet", \
                    process="preprocess", framework="tensorflow")
class BilinearImagenetTransform(BaseTransform):
    """Combination of a series of transforms which is applicable to images in Imagenet.

    Args:
        height: Height of the result
        width:Width of the result  
        central_fraction(float, default=0.875):fraction of size to crop
        mean_value(list, default=[0.0,0.0,0.0]):means for each channel
        scale(float, default=1.0):std value

    Returns:
        tuple of processed image and label
    """

    def __init__(self, height, width, central_fraction=0.875,
                 mean_value=[0.0,0.0,0.0], scale=1.0):

        self.height = height
        self.width = width
        self.mean_value = mean_value
        self.scale = scale
        self.central_fraction = central_fraction

    # sample is (images, labels)
    def __call__(self, sample):
        image, label = sample
        if image.dtype is not tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image containing 87.5% area of the original image.
        if self.central_fraction:
            image = tf.image.central_crop(image, central_fraction=self.central_fraction)

        if self.height and self.width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize(image, [self.height, self.width], \
                                    method=tf.image.ResizeMethod.BILINEAR)
            image = tf.squeeze(image, [0])

        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        means = tf.broadcast_to(self.mean_value, tf.shape(input=image))
        image = (image - means) * self.scale

        return (image, label)

@transform_registry(transform_type="BilinearImagenet", \
                    process="preprocess", framework="onnxrt_qlinearops, onnxrt_integerops")
class OnnxBilinearImagenetTransform(BaseTransform):
    """Combination of a series of transforms which is applicable to images in Imagenet.

    Args:
        height: Height of the result
        width:Width of the result  
        central_fraction(float, default=0.875):fraction of size to crop
        mean_value(list, default=[0.0,0.0,0.0]):means for each channel
        scale(float, default=1.0):std value

    Returns:
        tuple of processed image and label
    """

    def __init__(self, height, width, central_fraction=0.875,
                 mean_value=[0.0,0.0,0.0], scale=1.0):
        self.height = height
        self.width = width
        self.mean_value = mean_value
        self.scale = scale
        self.central_fraction = central_fraction

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, np.ndarray):
            image = image.astype('float32') / 255. 
        img_shape = image.shape
        depth = img_shape[2]
        img_hd = float(img_shape[0])
        bbox_h_start = int((img_hd - img_hd * self.central_fraction) / 2)
        img_wd = float(img_shape[1])
        bbox_w_start = int((img_wd - img_wd * self.central_fraction) / 2)

        bbox_h_size = img_shape[0] - bbox_h_start * 2
        bbox_w_size = img_shape[1] - bbox_w_start * 2

        image = image[bbox_h_start:bbox_h_start+bbox_h_size, bbox_w_start:bbox_w_start+bbox_w_size]

        if self.height and self.width:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        image = np.subtract(image, 0.5)
        image = np.multiply(image, 2.0)
        means = np.broadcast_to(self.mean_value, image.shape)
        image = (image - means) * self.scale
        image = image.astype(np.float32)

        return (image, label)

@transform_registry(transform_type="ResizeCropImagenet", \
                    process="preprocess", framework="onnxrt_qlinearops, onnxrt_integerops")
class ONNXResizeCropImagenetTransform(BaseTransform):
    """Combination of a series of transforms which is applicable to images in Imagenet.

    Args:
        height: Height of the result
        width:Width of the result  
        central_fraction(float, default=0.875):fraction of size to crop
        mean_value(list, default=[0.0,0.0,0.0]):means for each channel
        scale(float, default=1.0):std value

    Returns:
        tuple of processed image and label
    """

    def __init__(self, height, width, random_crop=False, resize_side=256, \
                 mean_value=[0.0,0.0,0.0], std_value=[0.229, 0.224, 0.225]):

        self.height = height
        self.width = width
        self.mean_value = mean_value
        self.std_value = std_value
        self.random_crop = random_crop
        self.resize_side = resize_side

    # sample is (images, labels)
    def __call__(self, sample):
        image, label = sample
        height, width = image.shape[0], image.shape[1]
        scale = self.resize_side / width if height > width else self.resize_side / height
        new_height = int(height*scale)
        new_width = int(width*scale)
        image = cv2.resize(image, (new_height, new_width))
        image = image / 255.
        shape = image.shape
        if self.random_crop:
            y0 = np.random.randint(low=0, high=(shape[0] - self.height +1))
            x0 = np.random.randint(low=0, high=(shape[1] - self.width +1))
        else:
            y0 = (shape[0] - self.height) // 2
            x0 = (shape[1] - self.width) // 2
        if len(image.shape) == 2:
            image = np.array([image])
            image = np.repeat(image, 3, axis=0)
            image = image.transpose(1, 2, 0)
        image = image[y0:y0+self.height, x0:x0+self.width, :]
        image = ((image - self.mean_value)/self.std_value).astype(np.float32)
        return (image.transpose(2, 0, 1), label)

@transform_registry(transform_type="ResizeWithAspectRatio", \
                    process="preprocess", framework="onnxrt_qlinearops, onnxrt_integerops")
class ResizeWithAspectRatio(BaseTransform):
    def __init__(self, height, width, scale=87.5, inter_pol=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.scale = scale
        self.inter_pol = inter_pol

    def __call__(self, sample):
        (img, label) = sample
        assert len(img.shape) == 3
        height, width, _ = img.shape
        new_height = int(100. * self.height / self.scale) 
        new_width = int(100. * self.width / self.scale)
        if height > width:
            w = new_width
            h = int(new_height * height / width)
        else:
            h = new_height
            w = int(new_width * width / height)
        img = cv2.resize(img, (w, h), interpolation=self.inter_pol)
        return img, label
