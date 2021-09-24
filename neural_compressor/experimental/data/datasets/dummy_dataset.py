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

from .dataset import dataset_registry, Dataset
import numpy as np
from neural_compressor.utils.utility import LazyImport
import logging

mx = LazyImport('mxnet')
torch = LazyImport('torch')

logger = logging.getLogger()

@dataset_registry(dataset_type="dummy", framework="tensorflow, tensorflow_itex, \
                                                    onnxrt_qlinearops, onnxrt_integerops, \
                                                    pytorch, pytorch_ipex, pytorch_fx, \
                                                    mxnet, engine",
                                                    dataset_format='')
class DummyDataset(Dataset):
    """Dataset used for dummy data generation.
       This Dataset is to construct a dataset from a specific shape.
       the value range is calculated from: low * stand_normal(0, 1) + high
       (TODO) construct dummy data from real dataset or iteration of data.

    Args: shape (list or tuple):support create multi shape tensors, use list of tuples
                                for each tuple in the list, will create a such size tensor.
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
          label (bool, default=False):whether to return 0 as label.
          transform (transform object, default=None): dummy dataset does not need transform.
                                                        If transform is not None, it will ignore
                                                        it.  
          filter (Filter objects, default=None): filter out examples according to 
                                                specific conditions

    """
    def __init__(self, shape, low=-128., high=127., dtype='float32', label=False, \
        transform=None, filter=None):

        dtype_map = {'float32':np.float32, 'float16':np.float16, 'uint8':np.uint8, \
                     'int8':np.int8, 'int32':np.int32, 'int64':np.int64, 'bool':np.bool}

        np.random.seed(9527)
        self.transform = transform
        self.label = label
        if isinstance(shape, list):
            # list tensor should same first demension n
            n = shape[0][0]
            assert all(isinstance(elem, tuple) and elem[0] == n for elem in shape), \
                'each tensor shape should be tuple and same fisrt demension'

            if isinstance(low, list):
                assert len(low) == len(shape) and all(isinstance(elem, float) for elem in low), \
                    'low list should have same length with shape with element data type float'
            else:
                low = (low * np.ones(len(shape))).astype(np.float)

            if isinstance(high, list):
                assert len(high) == len(shape) and all(isinstance(elem, float) for elem in high), \
                    'high list should have same length with shape with element data type float'
            else:
                high = (high * np.ones(len(shape))).astype(np.float)

            if isinstance(dtype, list):
                assert len(dtype) == len(shape) and \
                    all(elem in dtype_map.keys() for elem in dtype), \
                    'high list should have same length with shape with element data type float'
            else:
                dtype = [dtype for i in range(0, len(shape))]

        elif isinstance(shape, tuple):
            shape = [shape]
            if isinstance(low, float):
                low = [low]
            else:
                assert isinstance(low, list) and len(low) == 1 and isinstance(low[0], float), \
                    'low should be float or list of float with length 1'

            if isinstance(high, float):
                high = [high]
            else:
                assert isinstance(high, list) and len(high) == 1 and isinstance(high[0], float), \
                    'high should be float or list of float with length 1'

            if isinstance(dtype, str):
                assert dtype in dtype_map.keys(), 'dtype only support {}'.format(dtype_map.keys())
                dtype = [dtype]
            else:
                assert isinstance(dtype, list) and \
                    len(dtype) == 1 and dtype[0] in dtype_map.keys(), \
                    'dtype should be str or list of str in supported dtypes'

        self.dataset = []
        for idx in range(0, len(shape)):
            tensor = np.random.uniform(low=low[idx], high=high[idx], size=shape[idx])
            tensor = tensor.astype(dtype_map[dtype[idx]])
            self.dataset.append(tensor)

        if len(self.dataset) == 1:
            self.dataset = self.dataset[0]
        else:
            self.dataset = [elem for elem in zip(*self.dataset)]


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]
        if self.transform is not None:
            logger.warning("Dummy dataset does not need transform.")

        if self.label:
            return sample, 0
        else:
            return sample
