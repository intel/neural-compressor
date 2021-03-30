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
from .transform import transform_registry, BaseTransform
tf = LazyImport('tensorflow')
cv2 = LazyImport('cv2')

@transform_registry(transform_type="QuantizedInput", \
                    process="preprocess", framework="tensorflow")
class QuantizedInput(BaseTransform):
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
                    process="postprocess", framework="tensorflow")
class LabelShift(BaseTransform):
  def __init__(self, label_shift=0):
    self.label_shift = label_shift

  def __call__(self, sample):
    images, labels = sample
    labels = np.array(labels) - self.label_shift
    return images, labels
