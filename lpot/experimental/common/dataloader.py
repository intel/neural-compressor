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

from ..data import DATALOADERS

class DataLoader(object):
    """This class is just a wrapper of the information needed to build a dataloader,
       it can't yield batched data and only in this Quantization/Benchmark object's
       setter method a 'real' calib_dataloader will be created,
       the reason is we have to know the framework info
       and only after the Quantization/Benchmark object created then
       framework infomation can be known. Future we will support
       creating iterable dataloader from lpot.experimental.common.DataLoader
    """

    def __init__(self, dataset, batch_size=1, collate_fn=None,
                 last_batch='rollover', sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, shuffle=False):
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

def _generate_common_dataloader(dataloader, framework):
    if not isinstance(dataloader, DataLoader):
        assert hasattr(dataloader, '__iter__') and \
            hasattr(dataloader, 'batch_size'), \
            'dataloader must implement __iter__ method and batch_size attribute'
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
            shuffle=dataloader.shuffle)

