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
"""Definitions of the methods to fetch data from an iterable-style or list-style dataset."""

from abc import abstractmethod


class Fetcher(object):  # pragma: no cover
    """Base class for different fetchers."""

    def __init__(self, dataset, collate_fn, drop_last):
        """Initialize Fetcher.

        Args:
            dataset (object): dataset object from which to get data
            collate_fn (callable): merge data with outer dimension batch size
            drop_last (bool): whether to drop the last batch if it is incomplete
        """
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    @abstractmethod
    def __call__(self, batched_indices):
        """Fetch data.

        Args:
            batched_indices (list): fetch data according to batched_indices
        """
        raise NotImplementedError


class IterableFetcher(Fetcher):  # pragma: no cover
    """Iterate to get next batch-size samples as a batch."""

    def __init__(self, dataset, collate_fn, drop_last, distributed):
        """Initialize IterableFetcher.

        Args:
            dataset (object): dataset object from which to get data
            collate_fn (callable): merge data with outer dimension batch size
            drop_last (bool): whether to drop the last batch if it is incomplete
            distributed (bool): whether the dataloader is distributed
        """
        super(IterableFetcher, self).__init__(dataset, collate_fn, drop_last)
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


class IndexFetcher(Fetcher):  # pragma: no cover
    """Take single index or a batch of indices to fetch samples as a batch."""

    def __init__(self, dataset, collate_fn, drop_last, distributed):
        """Initialize IndexFetcher.

        Args:
            dataset (object): dataset object from which to get data
            collate_fn (callable): merge data with outer dimension batch size
            drop_last (bool): whether to drop the last batch if it is incomplete
            distributed (bool): whether the dataloader is distributed
        """
        super(IndexFetcher, self).__init__(dataset, collate_fn, drop_last)

    def __call__(self, batched_indices):
        """Fetch data.

        Args:
            batched_indices (list): fetch data according to batched_indices
        """
        data = [self.dataset[idx] for idx in batched_indices]
        return self.collate_fn(data)


FETCHERS = {
    "index": IndexFetcher,
    "iter": IterableFetcher,
}
