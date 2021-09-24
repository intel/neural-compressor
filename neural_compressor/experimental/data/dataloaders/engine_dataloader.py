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

from neural_compressor.utils.utility import LazyImport
from .base_dataloader import BaseDataLoader
from .default_dataloader import DefaultDataLoader
import numpy as np
import collections
import logging

def default_collate(batch):
    elem = batch[0]
    if isinstance(elem, collections.abc.Sequence):
        batch = zip(*batch)
        return [default_collate(samples) for samples in batch]
    elif isinstance(elem, dict):
        return elem
    else:
        try:
            return np.stack(batch)
        except:
            return batch

class EngineDataLoader(BaseDataLoader):
    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
                             sampler, batch_sampler, num_workers, pin_memory,
                             shuffle, distributed):
        if shuffle:
            logging.warning('Shuffle is not supported yet in EngineDataLoader, ' \
                            'ignoring shuffle keyword.')
        collate_fn = default_collate
        return DefaultDataLoader(dataset, batch_size, last_batch, collate_fn,
                                 sampler, batch_sampler, num_workers, pin_memory,
                                 shuffle, distributed)
