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
from lpot.utils.utility import LazyImport
from lpot.experimental.data.transforms import transform_registry, BaseTransform
tf = LazyImport('tensorflow')

# BELOW IS TO BE DEPRECATED!
@transform_registry(transform_type="ParseDecodeCoco", \
                    process="preprocess", framework="tensorflow")
class ParseDecodeCocoTransform(BaseTransform):
    def __call__(self, sample):
        # Dense features in Example proto.
        feature_map = {
            'image/encoded':
            tf.compat.v1.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/object/class/text':
            tf.compat.v1.VarLenFeature(dtype=tf.string),
            'image/object/class/label':
            tf.compat.v1.VarLenFeature(dtype=tf.int64),
            'image/source_id':tf.compat.v1.FixedLenFeature([], dtype=tf.string, default_value=''),
        }
        sparse_float32 = tf.compat.v1.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        feature_map.update({
            k: sparse_float32
            for k in [
                'image/object/bbox/xmin', 'image/object/bbox/ymin',
                'image/object/bbox/xmax', 'image/object/bbox/ymax'
            ]
        })

        features = tf.io.parse_single_example(sample, feature_map)

        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
        # Force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])

        encoded_image = features['image/encoded']
        image_tensor = tf.image.decode_image(encoded_image, channels=3)
        image_tensor.set_shape([None, None, 3])

        str_label = features['image/object/class/text'].values
        int_label = features['image/object/class/label'].values
        image_id = features['image/source_id']

        return image_tensor, (bbox[0], str_label, int_label, image_id)
