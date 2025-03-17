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

import collections
import math
import sys
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from neural_compressor.common import logger


def default_collate(batch):
    """Merge data with outer dimension batch size."""
    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        batch = zip(*batch)
        return [default_collate(samples) for samples in batch]
    elif isinstance(elem, np.ndarray):
        try:
            return np.stack(batch)
        except:
            return batch
    else:
        return batch


class IterableFetcher:
    """Iterate to get next batch-size samples as a batch."""

    def __init__(self, dataset, collate_fn, drop_last, distributed):
        """Initialize IterableFetcher.

        Args:
            dataset (object): dataset object from which to get data
            collate_fn (callable): merge data with outer dimension batch size
            drop_last (bool): whether to drop the last batch if it is incomplete
            distributed (bool): whether the dataloader is distributed
        """
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.dataset_iter = iter(dataset)
        self.index_whole = 0
        self.process_rank = 0  # The default rank is 0, which represents the main process
        self.process_size = 1  # By default, process_size=1, only the main process is running
        if distributed:
            import horovod.tensorflow as hvd

            hvd.init()
            self.process_rank = hvd.rank()
            self.process_size = hvd.size()
            if self.process_size < 2:
                raise EnvironmentError(
                    "The program is now trying to traverse"
                    " the distributed TensorFlow DefaultDataLoader in only one process."
                    " If you do not want to use distributed DataLoader, please set"
                    " 'distributed: False'. Or If you want to use distributed DataLoader,"
                    " please set 'distributed: True' and launch multiple processes."
                )

    def __call__(self, batched_indices):
        """Fetch data.

        Args:
            batched_indices (list): fetch data according to batched_indices
        """
        batch_data = []
        batch_size = len(batched_indices)
        while True:
            try:
                iter_data = next(self.dataset_iter)
                if (self.index_whole - self.process_rank) % self.process_size == 0:
                    batch_data.append(iter_data)
                self.index_whole += 1
                if len(batch_data) == batch_size:
                    break
            except StopIteration:
                break
        if len(batch_data) == 0 or (self.drop_last and len(batch_data) < len(batched_indices)):
            raise StopIteration
        return self.collate_fn(batch_data)


class IndexFetcher:
    """Take single index or a batch of indices to fetch samples as a batch."""

    def __init__(self, dataset, collate_fn, drop_last, distributed):
        """Initialize IndexFetcher.

        Args:
            dataset (object): dataset object from which to get data
            collate_fn (callable): merge data with outer dimension batch size
            drop_last (bool): whether to drop the last batch if it is incomplete
            distributed (bool): whether the dataloader is distributed
        """
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def __call__(self, batched_indices):
        """Fetch data.

        Args:
            batched_indices (list): fetch data according to batched_indices
        """
        data = [self.dataset[idx] for idx in batched_indices]
        return self.collate_fn(data)


class IterableSampler:
    """Internally samples elements.

    Used for datasets retrieved element by iterator. Yield None to act as a placeholder for each iteration.
    """

    def __init__(self, dataset):
        """Initialize IterableSampler.

        Args:
            dataset (object): dataset object from which to get data
        """
        self.whole_dataset = dataset

    def __iter__(self):
        """Yield data in iterative order."""
        while True:
            yield None

    def __len__(self):
        """Return the length of dataset."""
        return len(self.whole_dataset)


class SequentialSampler:
    """Sequentially samples elements, used for datasets retrieved element by index."""

    def __init__(self, dataset, distributed):
        """Initialize SequentialSampler.

        Args:
            dataset (object): dataset object from which to get data
            distributed (bool): whether the dataloader is distributed
        """
        self.whole_dataset = dataset
        self.distributed = distributed

    def __iter__(self):
        """Yield data in iterative order."""
        self.process_rank = 0  # The default rank is 0, which represents the main process
        self.process_size = 1  # By default, process_size=1, only the main process is running
        if self.distributed:
            import horovod.tensorflow as hvd

            hvd.init()
            self.process_rank = hvd.rank()
            self.process_size = hvd.size()
            if self.process_size < 2:
                raise EnvironmentError(
                    "The program is now trying to traverse"
                    " the distributed TensorFlow DefaultDataLoader in only one process."
                    " If you do not want to use distributed DataLoader, please set"
                    " 'distributed: False'. Or If you want to use distributed DataLoader,"
                    " please set 'distributed: True' and launch multiple processes."
                )
        return iter(range(self.process_rank, len(self.whole_dataset), self.process_size))

    def __len__(self):
        """Return the length of dataset."""
        return len(self.whole_dataset)


class BatchSampler:
    """Yield a batch of indices and number of batches."""

    def __init__(self, sampler, batch_size, drop_last=True):
        """Initialize BatchSampler.

        Args:
            sampler (Sampler): sampler used for generating batches
            batch_size (int): size of batch
            drop_last (bool, optional): whether to drop the last batch if it is incomplete. Defaults to True.
        """
        if isinstance(drop_last, bool):
            self.drop_last = drop_last
        else:
            raise ValueError("last_batch only support bool as input")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        """Yield data in iterative order."""
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        """Return the number of batches."""
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class BaseDataLoader:
    """Base class for TF DataLoaders.

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
        """Initialize DefaultDataLoader.

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
        self.last_batch = last_batch
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.drop_last = False if last_batch == "rollover" else True
        if self.collate_fn is None:
            self.collate_fn = default_collate

    def batch(self, batch_size, last_batch="rollover"):
        """Set batch_size and last_batch."""
        self.batch_size = batch_size
        self.last_batch = last_batch

    @property
    def dataloader(self):
        """Return dataloader."""
        return self

    def __iter__(self):
        """Yield data in iterative order."""
        return self._generate_dataloader(
            self.dataset,
            batch_size=self.batch_size,
            last_batch=self.last_batch,
            collate_fn=self.collate_fn,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            distributed=self.distributed,
        )

    def __len__(self):
        """Get dataset length."""
        try:
            dataset_len = self.dataset.__len__()
        except (AttributeError, TypeError):
            dataset_len = 0
            for _ in self.dataset:
                dataset_len += 1
        except Exception:
            raise ValueError(
                f"{self.dataset} is invalid, {self.dataset}"
                " does not support calculating the length of its dataloader"
            )
        if self.drop_last is False:
            dataloader_len = math.ceil(dataset_len / self.batch_size)
        else:
            dataloader_len = math.floor(dataset_len / self.batch_size)
        return dataloader_len

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
        sampler = self._generate_sampler(dataset, distributed)
        self.batch_sampler = BatchSampler(sampler, batch_size, self.drop_last)

        if self.dataset_type == "index":
            self.fetcher = IndexFetcher(dataset, collate_fn, self.drop_last, distributed)
        elif self.dataset_type == "iter":
            self.fetcher = IterableFetcher(dataset, collate_fn, self.drop_last, distributed)

        for batched_indices in self.batch_sampler:
            try:
                data = self.fetcher(batched_indices)
                yield data
            except StopIteration:
                return

    def _generate_sampler(self, dataset, distributed):
        if hasattr(dataset, "__getitem__"):
            self.dataset_type = "index"
            return SequentialSampler(dataset, distributed)
        elif hasattr(dataset, "__iter__"):
            self.dataset_type = "iter"
            return IterableSampler(dataset)
        else:
            raise ValueError("dataset type only support (index, iter)")


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
