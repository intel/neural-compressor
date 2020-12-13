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

from abc import abstractmethod


class BaseDataLoader(object):
    """Base class for all DataLoaders. _generate_dataloader is needed to create a dataloader object
       from the general params like batch_size and sampler. The dynamic batching is just to
       generate a new dataloader by setting batch_size and last_batch.

    """

    def __init__(self, dataset, batch_size=1, last_batch='rollover', collate_fn=None,
                 sampler=None, batch_sampler=None, num_workers=0, pin_memory=False):

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._batch_size = batch_size

        self.dataloader = self._generate_dataloader(
            self.dataset,
            batch_size=batch_size,
            last_batch=last_batch,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory)

    def batch(self, batch_size, last_batch='rollover'):
        self._batch_size = batch_size
        self.dataloader = self._generate_dataloader(
            self.dataset,
            batch_size,
            last_batch,
            self.collate_fn,
            self.sampler,
            self.batch_sampler,
            self.num_workers,
            self.pin_memory)

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        return iter(self.dataloader)

    @abstractmethod
    def _generate_dataloader(self, dataset, batch_size, last_batch,
                             collate_fn, sampler, batch_sampler, num_workers, pin_memory):
        raise NotImplementedError
