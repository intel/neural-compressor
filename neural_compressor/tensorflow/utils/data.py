#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""BaseDataloder of all dataloaders."""

import sys
from abc import abstractmethod

import numpy as np

from neural_compressor.common import logger


class BaseDataLoader:  # pragma: no cover
    """Base class for all DataLoaders.

    _generate_dataloader is needed to create a dataloader object
    from the general params like batch_size and sampler. The dynamic batching is just to
    generate a new dataloader by setting batch_size and last_batch.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        last_batch="rollover",
        collate_fn=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        distributed=False,
    ):
        """Initialize BaseDataLoader.

        Args:
            dataset (object): dataset from which to load the data
            batch_size (int, optional): number of samples per batch. Defaults to 1.
            last_batch (str, optional): whether to drop the last batch if it is incomplete.
                                        Support ['rollover', 'discard'], rollover means False, discard means True.
                                        Defaults to 'rollover'.
            collate_fn (callable, optional): merge data with outer dimension batch size. Defaults to None.
            sampler (Sampler, optional): Sampler object to sample data. Defaults to None.
            batch_sampler (BatchSampler, optional): BatchSampler object to generate batch of indices. Defaults to None.
            num_workers (int, optional): number of subprocesses to use for data loading. Defaults to 0.
            pin_memory (bool, optional): whether to copy data into pinned memory before returning. Defaults to False.
            shuffle (bool, optional): whether to shuffle data. Defaults to False.
            distributed (bool, optional): whether the dataloader is distributed. Defaults to False.
        """
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.last_batch = last_batch
        self.drop_last = False if last_batch == "rollover" else True

        self.dataloader = self._generate_dataloader(
            self.dataset,
            batch_size=batch_size,
            last_batch=last_batch,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            distributed=distributed,
        )

    def batch(self, batch_size, last_batch=None):
        """Set batch size for dataloader.

        Args:
            batch_size (int): number of samples per batch.
            last_batch (str, optional): whether to drop the last batch if it is incomplete.
                                        Support ['rollover', 'discard'], rollover means False, discard means True.
                                        Defaults to None.
        """
        self._batch_size = batch_size
        if last_batch is not None:
            self.last_batch = last_batch
        self.dataloader = self._generate_dataloader(
            self.dataset,
            batch_size,
            self.last_batch,
            self.collate_fn,
            self.sampler,
            self.batch_sampler,
            self.num_workers,
            self.pin_memory,
            self.shuffle,
            self.distributed,
        )

    @property
    def batch_size(self):
        """Get dataloader's batch_size.

        Returns:
            int: batch_size
        """
        return self._batch_size

    def __iter__(self):
        """Yield data in iterative order.

        Returns:
            iterator: iterator for dataloder
        """
        return iter(self.dataloader)

    @abstractmethod
    def _generate_dataloader(
        self,
        dataset,
        batch_size,
        last_batch,
        collate_fn,
        sampler,
        batch_sampler,
        num_workers,
        pin_memory,
        shuffle,
        distributed,
    ):
        raise NotImplementedError


class DummyDataset:  # pragma: no cover
    """Dataset used for dummy data generation.

    This Dataset is to construct a dataset from a specific shape.
    The value range is calculated from: low * stand_normal(0, 1) + high.
    (TODO) construct dummy data from real dataset or iteration of data.
    """

    def __init__(self, shape, low=-128.0, high=127.0, dtype="float32", label=True, transform=None, filter=None):
        """Initialize `DummyDataset` class.

        Args:
            shape (list or tuple): Support create multi shape tensors, use list of tuples
                for each tuple in the list, will create a such size tensor.
            low (list or float, default=-128.): Low out the tensor value range from [0, 1]
                to [0, low] or [low, 0] if low < 0, if float, will implement all tensors with same low value.
            high (list or float, default=127.): High the tensor value by add all tensor element
                value high. If list, length of list should be same with shape list.
            dtype (list or str, default='float32'): Support multi tensor dtype setting.
                If list, length of list should be same with shape list. If str, all tensors will
                use same dtype. dtype supports 'float32', 'float16', 'uint8', 'int8', 'int32', 'int64', 'bool'.
            label (bool, default=True): Whether to return 0 as label.
            transform (transform object, default=None): Dummy dataset does not need transform.
                If transform is not None, it will ignore it.
            filter (Filter objects, default=None): Filter out examples according to specific conditions.
        """
        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "uint8": np.uint8,
            "int8": np.int8,
            "int32": np.int32,
            "int64": np.int64,
            "bool": bool,
            "string": str,
        }

        np.random.seed(9527)
        self.transform = transform
        self.label = label
        if len(shape) == 0:
            logger.info("No data in the dummy dataset.")
        elif isinstance(shape, list):
            # list tensor should same first dimension n
            n = shape[0][0]
            assert all(
                isinstance(elem, tuple) and elem[0] == n for elem in shape
            ), "each tensor shape should be tuple and same first dimension"

            if isinstance(low, list):
                assert len(low) == len(shape) and all(
                    isinstance(elem, float) for elem in low
                ), "low list should have same length with shape with element data type float"
            else:
                low = (low * np.ones(len(shape))).astype(float)

            if isinstance(high, list):
                assert len(high) == len(shape) and all(
                    isinstance(elem, float) for elem in high
                ), "high list should have same length with shape with element data type float"
            else:
                high = (high * np.ones(len(shape))).astype(float)

            if isinstance(dtype, list):
                assert len(dtype) == len(shape) and all(
                    elem in dtype_map.keys() for elem in dtype
                ), "high list should have same length with shape with element data type float"
            else:
                dtype = [dtype for i in range(0, len(shape))]

        elif isinstance(shape, tuple):
            shape = [shape]
            if isinstance(low, float):
                low = [low]
            else:
                assert (
                    isinstance(low, list) and len(low) == 1 and isinstance(low[0], float)
                ), "low should be float or list of float with length 1"

            if isinstance(high, float):
                high = [high]
            else:
                assert (
                    isinstance(high, list) and len(high) == 1 and isinstance(high[0], float)
                ), "high should be float or list of float with length 1"

            if isinstance(dtype, str):
                assert dtype in dtype_map.keys(), "dtype only support {}".format(dtype_map.keys())
                dtype = [dtype]
            else:
                assert (
                    isinstance(dtype, list) and len(dtype) == 1 and dtype[0] in dtype_map.keys()
                ), "dtype should be str or list of str in supported dtypes"

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
        """Return the length of dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Return the item of dataset according to the given index."""
        sample = self.dataset[index]
        if self.transform is not None:
            logger.warning("Dummy dataset does not need transform.")

        if self.label:
            return sample, 0
        else:
            return sample


class DummyDatasetV2:  # pragma: no cover
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
