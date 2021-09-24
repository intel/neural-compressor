#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from .dataset import dataset_registry, IterableDataset
import numpy as np
from neural_compressor.utils.utility import LazyImport

mx = LazyImport('mxnet')
torch = LazyImport('torch')

@dataset_registry(dataset_type="dummy_v2", framework="tensorflow, tensorflow_itex, engine,\
                                                      onnxrt_qlinearops, onnxrt_integerops, \
                                                      pytorch, pytorch_ipex, pytorch_fx, mxnet",
                                                      dataset_format='')
class DummyDataset(IterableDataset):
    """Dataset used for dummy_v2 data generation.
       This Dataset is to construct a dataset from a input shape and label shape.
       the value range is calculated from: low * stand_normal(0, 1) + high

    Args: sample_size (int): total size of the dummy samples.
          input_shape (list or tuple): create single or multi input tensors, 
              tuple reperesent the sample shape of the dataset, eg an image size should be
              represented as (224, 224, 3), list contains multiple tuple and 
              represent multi input tensors.
          label_shape (list or tuple): create single or multi label tensors, 
              tuple reperesent the label shape of the dataset, eg an label size should be
              represented as (1, ), list contains multiple tuple and 
              represent multi label tensors.
          low (list or float, default=-128.):low out the tensor value range from [0, 1] 
                                            to [0, low] or [low, 0] if low < 0, if float, 
                                            will implement all tensors with same low value.  
          high (list or float, default=127.):high the tensor value by add all tensor element
                                            value high. If list, length of list should be 
                                            same with shape list.
          dtype (list or str, default='float32'):support multi tensor dtype setting. If list,
                                                length of list should be same with shape list,
                                                if str, all tensors will use same dtype. dtype
                                                support 'float32', 'float16', 'uint8', 'int8',
                                                'int32', 'int64', 'bool'.
          transform (transform object, default=None): dummy_v2 dataset does not need transform.
                                                        If transform is not None, it will ignore
                                                        it.  
          filter (Filter objects, default=None): filter out examples according to 
                                                specific conditions

    """
    def __init__(self, input_shape, label_shape=None, low=-128., high=127., \
                 dtype='float32', transform=None, filter=None):

        self.dtype_map = {'float32':np.float32, 'float16':np.float16, 'uint8':np.uint8, \
                     'int8':np.int8, 'int32':np.int32, 'int64':np.int64, 'bool':np.bool}

        np.random.seed(9527)
        self.transform = transform
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.low = low
        self.high = high
        self.dtype = dtype

        if label_shape is None:
            self.label_dim = 0
        elif isinstance(label_shape, tuple):
            self.label_dim = 1
        else:
            self.label_dim = len(label_shape)

        self.input_dim = 1 if isinstance(input_shape, tuple) else len(input_shape)
        self.total_dim = self.input_dim + self.label_dim

        if isinstance(high, list):
            assert len(high) == self.total_dim and \
                all(isinstance(elem, float) for elem in high),\
                'high value list length should same with label dim + input_dim'
        else:
            self.high = (high * np.ones(self.total_dim)).astype(np.float)

        if isinstance(low, list):
            assert len(low) == self.total_dim and \
                all(isinstance(elem, float) for elem in low), \
                'low value list length should same with label dim + input_dim'
        else:
            self.low = (low * np.ones(self.total_dim)).astype(np.float)

        if isinstance(dtype, list):
            assert len(dtype) == self.total_dim and \
                all(elem in self.dtype_map.keys() for elem in dtype), \
                'dtype list length should same with label dim + input_dim'
        else:
            self.dtype = [self.dtype for i in range(0, self.total_dim)]

        if isinstance(input_shape, tuple):
            self.input_shape = [input_shape]

        if isinstance(label_shape, tuple):
            self.label_shape = [label_shape]

    def __iter__(self):
        while True:
            input_data = []
            for idx in range(0, self.input_dim):
                tensor = np.random.uniform(\
                    low=self.low[idx], high=self.high[idx], size=self.input_shape[idx])
                tensor = tensor.astype(self.dtype_map[self.dtype[idx]])
                input_data.append(tensor)

            label = []
            for idx in range(0, self.label_dim):
                shift_idx = self.input_dim + idx 
                tensor = np.random.uniform(low=self.low[shift_idx],
                                           high=self.high[shift_idx],
                                           size=self.label_shape[idx])
                tensor = tensor.astype(self.dtype_map[self.dtype[shift_idx]])
                label.append(tensor)

            if len(input_data) == 1:
                input_data = input_data[0] 

            if len(label) == 1:
                label = label[0] 

            if len(label) > 0:
                yield input_data, label
            else:
                yield input_data
        
    def __len__(self):
        return sys.maxsize
