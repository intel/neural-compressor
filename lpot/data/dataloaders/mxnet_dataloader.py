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

import collections
import numpy as np
from lpot.utils.utility import LazyImport
from lpot.data.dataloaders.base_dataloader import BaseDataLoader
from lpot.data.dataloaders.default_dataloader import DefaultDataLoader
mx = LazyImport('mxnet')

def collate(batch):
    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        batch = zip(*batch)
        return [collate(samples) for samples in batch]
    elif isinstance(elem, np.ndarray):
        try:
            batch = [mx.nd.array(e) for e in batch]
            return mx.nd.stack(*batch)
        except:
            return np.stack(batch)
    elif isinstance(elem, mx.ndarray.NDArray): # pylint: disable=no-member
        return mx.nd.stack(*batch)
    else:
        return batch

class MXNetDataLoader(BaseDataLoader):
    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
                             sampler, batch_sampler, num_workers, pin_memory):
        drop_last = False if last_batch == 'rollover' else True
        if collate_fn is None:
            collate_fn = collate
        return DefaultDataLoader(dataset, batch_size, last_batch, collate_fn,
                                 sampler, batch_sampler, num_workers, pin_memory)
