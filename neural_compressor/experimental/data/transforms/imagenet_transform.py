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
from neural_compressor.utils.utility import LazyImport
from .transform import transform_registry, BaseTransform
tf = LazyImport('tensorflow')
cv2 = LazyImport('cv2')

@transform_registry(transform_type="QuantizedInput", \
                    process="preprocess", framework="tensorflow")
class QuantizedInput(BaseTransform):
    """Convert the dtype of input to quantize it.

    Args:
        dtype(str): desired image dtype, support 'uint8', 'int8'
        scale(float, default=None):scaling ratio of each point in image

    Returns:
        tuple of processed image and label
    """

    def __init__(self, dtype, scale=None):
        self.dtype_map = {'uint8': tf.uint8, 'int8': tf.int8}
        assert dtype in self.dtype_map.keys(), \
            'only support cast dtype {}'.format(self.dtype_map.keys())
        self.dtype = dtype
        self.scale = scale

    def __call__(self, sample):
        # scale is not know when tuning, in this case this transform
        # do nothing, it's only used when scale is set
        if self.scale == None:
            return sample
        image, label = sample
        image = image * self.scale
        if self.dtype == 'uint8':
            image = image + 128
        image = tf.dtypes.cast(image, dtype=self.dtype_map[self.dtype])
        return image, label

@transform_registry(transform_type="LabelShift", \
    process="postprocess", framework="tensorflow, onnxrt_qlinearops, \
    onnxrt_integerops, engine")
class LabelShift(BaseTransform):
    """Convert label to label - label_shift.

    Args:
        label_shift(int, default=0): number of label shift

    Returns:
        tuple of processed image and label
    """

    def __init__(self, label_shift=0):
        self.label_shift = label_shift

    def __call__(self, sample):
        images, labels = sample
        if isinstance(labels, np.ndarray):
            labels = labels - self.label_shift
        elif isinstance(labels, list):
            if isinstance(labels[0], tuple):
                labels = [tuple(np.array(label) - self.label_shift) for label in labels]
            elif isinstance(labels[0], np.ndarray):
                labels = [label - self.label_shift for label in labels]
            else:
                labels = np.array(labels) - self.label_shift
                labels = labels.tolist()
        else:
            labels = np.array(labels) - self.label_shift
        return images, labels

class ParseDecodeImagenet():
    """Parse features in Example proto.

    Returns:
        tuple of parsed image and label
    """

    def __call__(self, sample):
        # Dense features in Example proto.
        feature_map = {
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/class/label': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=-1)}

        sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        feature_map.update(
            {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                         'image/object/bbox/ymin',
                                         'image/object/bbox/xmax',
                                         'image/object/bbox/ymax']})

        features = tf.io.parse_single_example(serialized=sample, features=feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)
        image = features['image/encoded']
        image = tf.image.decode_jpeg(
            image, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')

        return (image, label)
