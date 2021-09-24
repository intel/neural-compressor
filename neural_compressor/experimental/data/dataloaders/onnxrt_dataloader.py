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
from ..datasets.bert_dataset import ONNXRTBertDataset
import logging
torch = LazyImport('torch')

class ONNXRTBertDataLoader(DefaultDataLoader):
    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
                             sampler, batch_sampler, num_workers, pin_memory,
                             shuffle, distributed):
        import numpy as np
        from torch.utils.data import DataLoader, SequentialSampler
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, \
                                batch_size=batch_size)
        dynamic_length = dataset.dynamic_length
        model_type = dataset.model_type
        max_seq_length = dataset.max_seq_length

        for batch in dataloader:
            try:
                batch_seq_length = max_seq_length if not dynamic_length \
                    else torch.max(batch[-2], 0)[0].item()
                batch = tuple(t.detach().cpu().numpy()  \
                            if not isinstance(t, np.ndarray) else t \
                            for t in batch)
                if model_type == 'bert':
                    data = [
                        batch[0][:,:batch_seq_length],
                        batch[1][:,:batch_seq_length],
                        batch[2][:,:batch_seq_length]
                    ]
                else:
                    data = [
                        batch[0][:,:batch_seq_length],
                        batch[1][:,:batch_seq_length]
                    ]
                label = batch[-1]
                yield data, label
            except StopIteration:
                return

class ONNXRTDataLoader(BaseDataLoader):
    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
                             sampler, batch_sampler, num_workers, pin_memory,
                             shuffle, distributed):
        if shuffle:
            logging.warning('Shuffle is not supported yet in ONNXRTDataLoader, ' \
                            'ignoring shuffle keyword.')

        if isinstance(dataset, ONNXRTBertDataset):
            return ONNXRTBertDataLoader(dataset, batch_size, last_batch, collate_fn,
                                     sampler, batch_sampler, num_workers, pin_memory,
                                     shuffle, distributed)
        else:
            return DefaultDataLoader(dataset, batch_size, last_batch, collate_fn,
                                     sampler, batch_sampler, num_workers, pin_memory,
                                     shuffle, distributed)
