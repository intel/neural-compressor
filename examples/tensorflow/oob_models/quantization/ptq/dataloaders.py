#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

from neural_compressor.experimental.data.dataloaders.fetcher import FETCHERS
from neural_compressor.experimental.data.dataloaders.sampler import BatchSampler
from neural_compressor.experimental.data.dataloaders.default_dataloader import DefaultDataLoader

# special dataloader for oob wide_deep model
class WidedeepDataloader(DefaultDataLoader):

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn, sampler,
                            batch_sampler, num_workers, pin_memory, shuffle, distributed):

        sampler = self._generate_sampler(dataset, distributed)
        self.batch_sampler = BatchSampler(sampler, batch_size, self.drop_last)
        self.fetcher = FETCHERS[self.dataset_type](dataset, collate_fn, self.drop_last, distributed)

        for batched_indices in self.batch_sampler:
            try:
                data = self.fetcher(batched_indices)
                data[0][0] = data[0][0][0]
                yield data
            except StopIteration:
                return
