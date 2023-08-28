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
"""Initialize the DATASETS class."""

import numpy as np

from neural_compressor.utils.utility import LazyImport

from .base_dataloader import BaseDataLoader

torch = LazyImport("torch")
hvd = LazyImport("horovod.torch")


class PyTorchDataLoader(BaseDataLoader):  # pragma: no cover
    """PyTorchDataLoader inherits from BaseDataLoader."""

    def _generate_dataloader(
        self,
        dataset,
        batch_size,
        last_batch,
        collate_fn,
        sampler,
        batch_sampler,
        num_workers,
        pin_memory,
        shuffle,
        distributed,
    ):
        """Generate PyTorch dataloader.

        Args:
            dataset: dataset
            batch_size (int): batch size
            last_batch (string): rollover last batch or not.
            collate_fn: collate_fn
            sampler: sampler
            batch_sampler: batch_sampler
            num_workers (int): num_workers
            pin_memory (bool): pin_memory
            shuffle (bool): shuffle
            distributed (bool): distributed

        Returns:
            _type_: _description_
        """
        drop_last = False if last_batch == "rollover" else True
        assert len(dataset) != 0, "Warning: Dataset is empty, Please check dataset path!"
        if distributed and sampler is None:
            # TODO: lazy init here
            hvd.init()
            # sampler option is mutually exclusive with shuffle pytorch
            self.sampler = sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank()
            )

        return torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            batch_sampler=batch_sampler,
        )
