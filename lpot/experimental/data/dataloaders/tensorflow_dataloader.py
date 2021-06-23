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

from lpot.utils.utility import LazyImport
from abc import abstractmethod
import collections
import numpy as np
from .sampler import IterableSampler, SequentialSampler, BatchSampler
from .fetcher import FETCHERS
from .default_dataloader import default_collate
from .default_dataloader import DefaultDataLoader
from ..datasets.bert_dataset import TensorflowBertDataset
from .base_dataloader import BaseDataLoader
import logging

tf = LazyImport('tensorflow')
lpot = LazyImport('lpot')

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
                             num_workers=None, pin_memory=None, shuffle=False):
        drop_last = False if last_batch == 'rollover' else True
        if shuffle:
            logging.warning('Shuffle is not supported yet in TFDataLoader, ' \
                            'ignoring shuffle keyword.')

        def check_dynamic_shape(element_spec):
            if isinstance(element_spec, collections.abc.Sequence):
                return any([check_dynamic_shape(ele) for ele in element_spec])
            elif isinstance(element_spec, tf.TensorSpec):
                return True if element_spec.shape.num_elements() is None else False
            else:
                raise ValueError('unrecognised element spec...')

        def squeeze_output(output):
            if isinstance(output, collections.abc.Sequence):
                return [squeeze_output(ele) for ele in output]
            elif isinstance(output, np.ndarray):
                return np.squeeze(output, axis=0)
            else:
                raise ValueError('not supported output format....')

        try_single_batch = check_dynamic_shape(dataset.element_spec)
        dataset = dataset.batch(1 if try_single_batch else batch_size, drop_last)
        if tf.executing_eagerly():
            for iter_tensors in dataset:
                yield [elem.numpy() for elem in iter_tensors]
        else:
            ds_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            iter_tensors = ds_iterator.get_next()
            data_config = tf.compat.v1.ConfigProto()
            data_config.use_per_session_threads = 1
            data_config.intra_op_parallelism_threads = 1
            data_config.inter_op_parallelism_threads = 16
            data_sess = tf.compat.v1.Session(config=data_config)
            # pylint: disable=no-name-in-module
            from tensorflow.python.framework.errors_impl import OutOfRangeError
            while True:
                try:
                    if not try_single_batch:
                        outputs = data_sess.run(iter_tensors)
                        yield outputs
                    else:
                        outputs = [squeeze_output(data_sess.run(iter_tensors)) \
                            for i in range(0, batch_size) ]
                        outputs = default_collate(outputs)
                        yield outputs
                except OutOfRangeError:
                    data_sess.close()
                    return

class TensorflowBertDataLoader(DefaultDataLoader):
    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
                             sampler, batch_sampler, num_workers, pin_memory, shuffle):

        if shuffle:
            logging.warning('Shuffle is not supported yet in TensorflowBertDataLoader, ' \
                            'ignoring shuffle keyword.')
        def bert_collate_fn(batch):
            elem = batch[0]
            return elem
        drop_last = False if last_batch == 'rollover' else True
        sampler = self._generate_sampler(dataset)
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.fetcher = FETCHERS[self.dataset_type](dataset, bert_collate_fn, drop_last)

        for batched_indices in self.batch_sampler:
            try:
                data = self.fetcher(batched_indices)
                yield (data[0], batch_size), data[1]
            except StopIteration:
                return

class TensorflowDataLoader(BaseDataLoader):
    """DataLoader for frameework Tensorflow, if it's a tf.data.Dataset we will directly use
       the dataloader in the other case will use DefaultDataLoader instead.

    """

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn, \
                sampler, batch_sampler, num_workers, pin_memory, shuffle):

        if shuffle:
            logging.warning('Shuffle is not supported yet in TensorflowDataLoader, ' \
                            'ignoring shuffle keyword.')
        if isinstance(dataset, tf.data.Dataset):
            return TFDataDataLoader(dataset, batch_size, last_batch=last_batch)
        elif isinstance(dataset, TensorflowBertDataset):
            return TensorflowBertDataLoader(dataset, batch_size, last_batch,
                        collate_fn, sampler, batch_sampler, num_workers,
                        pin_memory, shuffle)
        else:
            return DefaultDataLoader(dataset, batch_size, last_batch, collate_fn,
                                     sampler, batch_sampler, num_workers,
                                     pin_memory, shuffle)
