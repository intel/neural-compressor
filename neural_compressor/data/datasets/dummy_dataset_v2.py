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
# ==============================================================================
"""Dummy dataset for dummy_v2/sparse_dummy_v2 data generation on multiple framework backends."""

import sys
from functools import reduce

import numpy as np

from neural_compressor.utils.utility import LazyImport

from .dataset import IterableDataset, dataset_registry

mx = LazyImport("mxnet")
torch = LazyImport("torch")


@dataset_registry(
    dataset_type="dummy_v2",
    framework="tensorflow, tensorflow_itex, \
                                                      onnxrt_qlinearops, onnxrt_integerops, \
                                                      pytorch, pytorch_ipex, pytorch_fx, mxnet",
    dataset_format="",
)
class DummyDataset(IterableDataset):  # pragma: no cover
    """Dataset used for dummy_v2 data generation.

    This Dataset is to construct a dataset from a input shape and label shape.
    The value range is calculated from: low * stand_normal(0, 1) + high.
    """

    def __init__(
        self, input_shape, label_shape=None, low=-128.0, high=127.0, dtype="float32", transform=None, filter=None
    ):
        """Initialize `DummyDataset` class.

        Args:
            sample_size (int): Total size of the dummy samples.
            input_shape (list or tuple): Create single or multi input tensors,
                tuple represent the sample shape of the dataset, e.g. an image size should be
                represented as (224, 224, 3), list contains multiple tuple and represent multi input tensors.
            label_shape (list or tuple): Create single or multi label tensors,
                tuple represent the label shape of the dataset, e.g. an label size should be
                represented as (1, ), list contains multiple tuple and represent multi label tensors.
            low (list or float, default=-128.): Low out the tensor value range from [0, 1]
                to [0, low] or [low, 0] if low < 0. If float, will implement all tensors with same low value.
            high (list or float, default=127.): High the tensor value by add all tensor element value high.
                If list, length of list should be same with shape list.
            dtype (list or str, default='float32'): Support multi tensor dtype setting.
                If list, length of list should be same with shape list.
                If str, all tensors will use same dtype.
                dtype supports 'float32', 'float16', 'uint8', 'int8','int32', 'int64', 'bool'.
            transform (transform object, default=None): dummy_v2 dataset does not need transform.
                If transform is not None, it will ignore it.
            filter (Filter objects, default=None): Filter out examples according to specific conditions.
        """
        self.dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "uint8": np.uint8,
            "int8": np.int8,
            "int32": np.int32,
            "int64": np.int64,
            "bool": bool,
        }

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
            assert len(high) == self.total_dim and all(
                isinstance(elem, float) for elem in high
            ), "high value list length should same with label dim + input_dim"
        else:
            self.high = (high * np.ones(self.total_dim)).astype(np.float32)

        if isinstance(low, list):
            assert len(low) == self.total_dim and all(
                isinstance(elem, float) for elem in low
            ), "low value list length should same with label dim + input_dim"
        else:
            self.low = (low * np.ones(self.total_dim)).astype(np.float32)

        if isinstance(dtype, list):
            assert len(dtype) == self.total_dim and all(
                elem in self.dtype_map.keys() for elem in dtype
            ), "dtype list length should same with label dim + input_dim"
        else:
            self.dtype = [self.dtype for i in range(0, self.total_dim)]

        if isinstance(input_shape, tuple):
            self.input_shape = [input_shape]

        if isinstance(label_shape, tuple):
            self.label_shape = [label_shape]

    def __iter__(self):
        """Yield data in iterative order."""
        while True:
            input_data = []
            for idx in range(0, self.input_dim):
                tensor = np.random.uniform(low=self.low[idx], high=self.high[idx], size=self.input_shape[idx])
                tensor = tensor.astype(self.dtype_map[self.dtype[idx]])
                input_data.append(tensor)

            label = []
            for idx in range(0, self.label_dim):
                shift_idx = self.input_dim + idx
                tensor = np.random.uniform(
                    low=self.low[shift_idx], high=self.high[shift_idx], size=self.label_shape[idx]
                )
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
        """Return the length of dataset."""
        return sys.maxsize


@dataset_registry(
    dataset_type="sparse_dummy_v2",
    framework="tensorflow, tensorflow_itex, \
                                                      onnxrt_qlinearops, onnxrt_integerops, \
                                                      pytorch, pytorch_ipex, pytorch_fx, mxnet",
    dataset_format="",
)
class SparseDummyDataset(IterableDataset):  # pragma: no cover
    """Dataset used for sparse_dummy_v2 data generation.

    This Dataset is to construct a dataset from a input shape and label shape.
    The value range is calculated from: low * stand_normal(0, 1) + high.
    """

    def __init__(
        self,
        dense_shape,
        label_shape=None,
        sparse_ratio=0.5,
        low=-128.0,
        high=127.0,
        dtype="float32",
        transform=None,
        filter=None,
    ):
        """Initialize `SparseDummyDataset` class.

        Args:
            sample_size (int): Total size of the dummy samples.
            dense_shape (list or tuple): Create single or multi sparse tensors, tuple represent
                the sample shape of the dataset, e.g. an image size should be represented as (224, 224, 3),
                list contains multiple tuple and represent multi input tensors.
            label_shape (list or tuple): Create single or multi label tensors, tuple represent
                the label shape of the dataset, e.g. an label size should be represented as (1, ),
                list contains multiple tuple and represent multi label tensors.
            sparse_ratio (float, default=0.5): The ratio of sparsity, supports [0, 1].
            low (list or float, default=-128.): Low out the tensor value range from [0, 1]
                to [0, low] or [low, 0] if low < 0. If float, will implement all tensors with same low value.
            high (list or float, default=127.): High the tensor value by add all tensor element value high.
                If list, length of list should be same with shape list.
            dtype (list or str, default='float32'): Support multi tensor dtype setting. If list,
                length of list should be same with shape list. If str, all tensors will use same dtype.
                dtype supports 'float32', 'float16', 'uint8', 'int8', 'int32', 'int64', 'bool'.
            transform (transform object, default=None): dummy_v2 dataset does not need transform.
                If transform is not None, it will ignore it.
            filter (Filter objects, default=None): Filter out examples according to specific conditions.
        """
        self.dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "uint8": np.uint8,
            "int8": np.int8,
            "int32": np.int32,
            "int64": np.int64,
            "bool": bool,
        }

        np.random.seed(9527)
        self.transform = transform
        self.dense_shape = dense_shape
        self.label_shape = label_shape
        self.sparse_ratio = sparse_ratio
        self.low = low
        self.high = high
        self.dtype = dtype

        if isinstance(dense_shape, tuple):
            self.dense_shape = [dense_shape]

        if label_shape is None:
            self.label_dim = 0
        else:
            if isinstance(label_shape, tuple):
                self.label_shape = [label_shape]
            if len(self.label_shape) == 1 and len(self.label_shape) != len(self.dense_shape):
                self.label_shape = len(self.dense_shape) * self.label_shape
            assert len(self.label_shape) == len(
                self.dense_shape
            ), "length of dense_shape should be euqal to length of label_shape"
            self.label_dim = len(self.label_shape)

        self.input_dim = 1 if isinstance(dense_shape, tuple) else len(dense_shape)
        self.total_dim = self.input_dim + self.label_dim

        if isinstance(sparse_ratio, list):
            assert len(sparse_ratio) == self.input_dim and all(
                isinstance(elem, float) for elem in sparse_ratio
            ), "sparse_ratio list length should same with input_dim"
        else:
            self.sparse_ratio = (sparse_ratio * np.ones(self.input_dim)).astype(np.float32)
        assert all([0 <= i <= 1 for i in self.sparse_ratio]), "sparse_ratio should be in [0,1]"

        if isinstance(high, list):
            assert len(high) == self.total_dim and all(
                isinstance(elem, float) for elem in high
            ), "high value list length should same with label dim + input_dim"
        else:
            self.high = (high * np.ones(self.total_dim)).astype(np.float32)

        if isinstance(low, list):
            assert len(low) == self.total_dim and all(
                isinstance(elem, float) for elem in low
            ), "low value list length should same with label dim + input_dim"
        else:
            self.low = (low * np.ones(self.total_dim)).astype(np.float32)

        if isinstance(dtype, list):
            assert len(dtype) == self.total_dim and all(
                elem in self.dtype_map.keys() for elem in dtype
            ), "dtype list length should same with label dim + input_dim"
        else:
            self.dtype = [self.dtype for i in range(0, self.total_dim)]

    def __iter__(self):
        """Yield data in iterative order."""
        while True:
            input_data = []
            for idx, shape in enumerate(self.dense_shape):
                dim = len(shape)
                total = reduce(lambda x, y: x * y, shape)
                sparse_num = round(total * (1 - self.sparse_ratio[idx]))
                val = np.random.uniform(low=self.low[idx], high=self.high[idx], size=sparse_num)
                val = val.astype(self.dtype_map[self.dtype[idx]])
                nums = np.arange(sparse_num)
                indices = []
                dim_shape = [reduce(lambda x, y: x * y, shape[i:]) / shape[i] for i in range(len(shape))]
                for num in nums:
                    indice = []
                    for item in dim_shape:
                        indice.append(num // item)
                        num = num - indice[-1] * item if num - indice[-1] * item > 0 else num
                    indices.append(indice)

                if self.label_dim > 0:
                    shift_idx = self.input_dim + idx
                    tensor = np.random.uniform(
                        low=self.low[shift_idx], high=self.high[shift_idx], size=self.label_shape[idx]
                    )
                    tensor = tensor.astype(self.dtype_map[self.dtype[shift_idx]])
                    input_data.append([(np.array(indices), val), tensor])
                else:
                    input_data.append((np.array(indices), val))

            yield input_data

    def __len__(self):
        """Return the length of dataset."""
        return sys.maxsize
