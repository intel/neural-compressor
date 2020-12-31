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
from .default_dataloader import DefaultDataLoader
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
        assert not tf.executing_eagerly(), 'tf dataloader not allowed executed in eager mode...'
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
        
