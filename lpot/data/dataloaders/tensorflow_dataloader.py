#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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

from lpot.utils.utility import LazyImport
from abc import abstractmethod
import collections
import numpy as np
from .sampler import IterableSampler, SequentialSampler, BatchSampler
from .base_dataloader import BaseDataLoader

tf = LazyImport('tensorflow')


class TensorflowDataLoader(BaseDataLoader):
    """DataLoader for frameework Tensorflow, if it's a tf.data.Dataset we will directly use
       the dataloader in the other case will use DefaultDataLoader instead.

    """

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
                             sampler, batch_sampler, num_workers, pin_memory):

        if isinstance(dataset, tf.data.Dataset):
            return TFDataDataLoader(dataset, batch_size, last_batch=last_batch)
        else:
            return DefaultDataLoader(dataset, batch_size, last_batch, collate_fn,
                                     sampler, batch_sampler, num_workers, pin_memory)

class TFDataDataLoader(BaseDataLoader):
    """In tensorflow1.x dataloader is coupled with the graph, but it also support feed_dict
       method to do session run, this dataloader is designed to satisfy the usage of feed dict
       in tf1.x. Although it's a general dataloader and can be used in MXNet and PyTorch.

    """

    def __init__(self, dataset, batch_size=1, last_batch='rollover'):

        self.dataset = dataset
        self.last_batch = last_batch
        self._batch_size = batch_size
        dataset = dataset.batch(batch_size)
        
    def batch(self, batch_size, last_batch='rollover'):
        drop_last = False if last_batch == 'rollover' else True
        self._batch_size = batch_size
        self.dataset = self.dataset.batch(batch_size, drop_last)

    def __iter__(self):
        return self._generate_dataloader(
            self.dataset,
            batch_size=self.batch_size,
            last_batch=self.last_batch,)

    def _generate_dataloader(self, dataset, batch_size=1, last_batch='rollover', \
                             collate_fn=None, sampler=None, batch_sampler=None, \
                             num_workers=None, pin_memory=None):

        drop_last = False if last_batch == 'rollover' else True
        dataset = dataset.batch(batch_size, drop_last)
        ds_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        iter_tensors = ds_iterator.get_next()
        data_config = tf.compat.v1.ConfigProto() 
        data_config.use_per_session_threads = 1
        data_config.intra_op_parallelism_threads = 1
        data_config.inter_op_parallelism_threads = 16
        data_sess = tf.compat.v1.Session(config=data_config)
        from tensorflow.python.framework.errors_impl import OutOfRangeError
        while True:
            try:
                outputs = data_sess.run(iter_tensors)
                yield outputs
            except OutOfRangeError:
                data_sess.close()
                return
        
def default_collate(batch):
    """Puts each data field into a pd frame with outer dimension batch size"""
    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        batch = zip(*batch)
        return [default_collate(samples) for samples in batch]
    elif isinstance(elem, np.ndarray):
        return np.stack(batch)
    else:
        return batch


class Fetcher(object):
    def __init__(self, dataset, collate_fn, drop_last):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    @abstractmethod
    def __call__(self, batched_indices):
        raise NotImplementedError


class IterableFetcher(Fetcher):
    def __init__(self, dataset, collate_fn, drop_last):
        super(IterableFetcher, self).__init__(dataset, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def __call__(self, batched_indices):
        data = []
        for _ in batched_indices:
            try:
                data.append(next(self.dataset_iter))
            except StopIteration:
                break
        if len(data) == 0 or (self.drop_last and len(data) < len(batched_indices)):
            raise StopIteration
        return self.collate_fn(data)


class IndexFetcher(Fetcher):
    def __init__(self, dataset, collate_fn, drop_last):
        super(IndexFetcher, self).__init__(dataset, collate_fn, drop_last)

    def __call__(self, batched_indices):
        data = [self.dataset[idx] for idx in batched_indices]

        return self.collate_fn(data)


FETCHERS = {"index": IndexFetcher, "iter": IterableFetcher, }


class DefaultDataLoader(BaseDataLoader):
    """In tensorflow1.x dataloader is coupled with the graph, but it also support feed_dict
       method to do session run, this dataloader is designed to satisfy the usage of feed dict
       in tf1.x. Although it's a general dataloader and can be used in MXNet and PyTorch.

    """

    def __init__(self, dataset, batch_size=1, last_batch='rollover', collate_fn=None,
                 sampler=None, batch_sampler=None, num_workers=0, pin_memory=False):

        self.dataset = dataset
        self.last_batch = last_batch
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self._batch_size = batch_size
        if self.collate_fn == None:
            self.collate_fn = default_collate
        
    def batch(self, batch_size, last_batch='rollover'):
        self._batch_size = batch_size
        self.last_batch = last_batch

    @property
    def dataloader(self):
        return self

    def __iter__(self):
        return self._generate_dataloader(
            self.dataset,
            batch_size=self.batch_size,
            last_batch=self.last_batch,
            collate_fn=self.collate_fn,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)

    def _generate_sampler(self, dataset):
        if hasattr(dataset, "__getitem__"):
            self.dataset_type = 'index'
            return SequentialSampler(self.dataset)
        elif hasattr(dataset, "__iter__"):
            self.dataset_type = 'iter'
            return IterableSampler()
        else:
            raise ValueError("dataset type only support (index, iter)")

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
                             sampler, batch_sampler, num_workers, pin_memory):

        drop_last = False if last_batch == 'rollover' else True
        sampler = self._generate_sampler(dataset)
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.fetcher = FETCHERS[self.dataset_type](dataset, collate_fn, drop_last)

        for batched_indices in self.batch_sampler:
            try:
                data = self.fetcher(batched_indices)
                yield data
            except StopIteration:
                return
