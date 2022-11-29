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

"""common DataLoader just collects the information to construct a dataloader."""

from ..data import DATALOADERS

class DataLoader(object):
    """A wrapper of the information needed to construct a dataloader.

    This class can't yield batched data and only in this Quantization/Benchmark 
    object's setter method a 'real' calib_dataloader will be created, the reason 
    is we have to know the framework info and only after the Quantization/Benchmark
    object created then framework infomation can be known. Future we will support
    creating iterable dataloader from neural_compressor.experimental.common.DataLoader
    """

    def __init__(self, dataset, batch_size=1, collate_fn=None,
                 last_batch='rollover', sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, shuffle=False, distributed=False):
        """Initialize a Dataloader with needed information.

        Args:
            dataset (object): A dataset object from which to get data. Dataset must implement 
                __iter__ or __getitem__ method.
            batch_size (int, optional): How many samples per batch to load. Defaults to 1.
            collate_fn (Callable, optional): Callable function that processes the batch you 
                want to return from your dataloader. Defaults to None.
            last_batch (str, optional): How to handle the last batch if the batch size does 
                not evenly divide by the number of examples in the dataset. 'discard': throw 
                it away. 'rollover': insert the examples to the beginning of the next batch.
                Defaults to 'rollover'.
            sampler (Iterable, optional): Defines the strategy to draw samples from the dataset.
                Defaults to None.
            batch_sampler (Iterable, optional): Returns a batch of indices at a time. Defaults to None.
            num_workers (int, optional): how many subprocesses to use for data loading. 
                0 means that the data will be loaded in the main process. Defaults to 0.
            pin_memory (bool, optional): If True, the data loader will copy Tensors into device 
                pinned memory before returning them. Defaults to False.
            shuffle (bool, optional): Set to ``True`` to have the data reshuffled
                at every epoch. Defaults to False.
            distributed (bool, optional): Set to ``True`` to support distributed computing. 
                Defaults to False.
        """
        assert hasattr(dataset, '__iter__') or \
               hasattr(dataset, '__getitem__'), \
               "dataset must implement __iter__ or __getitem__ magic method!"
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.last_batch = last_batch
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.distributed = distributed

def _generate_common_dataloader(dataloader, framework, distributed=False):
    """Generate common dataloader.
    
    Args:
        dataloader (generator): A dataloader which can yield tuple of (input, label)/(input, _) 
            batched data.
        framework (str): The string of supported framework.
        distributed (bool, optional): Set to ``True`` to support distributed computing. 
            Defaults to False.
            
    Returns:
        BaseDataLoader: neural_compressor built-in dataloader
    """
    if not isinstance(dataloader, DataLoader):
        assert hasattr(dataloader, '__iter__') and \
            hasattr(dataloader, 'batch_size'), \
            'dataloader must implement __iter__ method and batch_size attribute'
        assert not distributed, "Please use \
            neural_compressor.experimental.common.DataLoader to support distributed computing"
        return dataloader
    else:
        return DATALOADERS[framework](
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            collate_fn=dataloader.collate_fn,
            last_batch=dataloader.last_batch,
            sampler=dataloader.sampler,
            batch_sampler=dataloader.batch_sampler,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            shuffle=dataloader.shuffle,
            distributed=bool(dataloader.distributed or distributed))

