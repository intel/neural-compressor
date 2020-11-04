#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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
from ilit.utils.utility import LazyImport
from .transform import transform_registry, Transform
tf = LazyImport('tensorflow')

@transform_registry(transform_type="ParseDecodeCoco", \
                    process="preprocess", framework="tensorflow")
class ParseDecodeCocoTransform(Transform):
    def __call__(self, sample):
        # Dense features in Example proto.
        feature_map = {
            'image/encoded': 
            tf.compat.v1.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/object/class/text':
            tf.compat.v1.VarLenFeature(dtype=tf.string),
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

        label = features['image/object/class/text'].values
        image_id = features['image/source_id']

        return image_tensor, (bbox[0], label, image_id)

@transform_registry(transform_type="COCOPreds",\
                    process="postprocess", framework="tensorflow")
class COCOPreds(Transform):
    def __call__(self, sample):
        from ilit.metric.coco_label_map import category_map
        category_map_reverse = {v: k for k, v in category_map.items()}
        
        preds = sample[0]
        bbox = sample[1][0]
        label = sample[1][1]
        image_id = sample[1][2]

        detection = {}
        num = int(preds[0][0])
        detection['boxes'] = np.asarray(preds[1][0])[0:num]
        detection['scores'] = np.asarray(preds[2][0])[0:num]
        detection['classes'] = np.asarray(preds[3][0])[0:num]

        ground_truth = {}
        ground_truth['boxes'] = np.asarray(bbox[0])
        label = [
            x if type(x) == 'str' else x.decode('utf-8')
            for x in label[0]
        ]
        ground_truth['classes'] = np.asarray(
            [category_map_reverse[x] for x in label])
        image_id = image_id[0] if type(
            image_id[0]) == 'str' else image_id[0].decode('utf-8')


        return detection, (ground_truth, image_id)
