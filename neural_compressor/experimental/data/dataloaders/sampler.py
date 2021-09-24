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

from abc import abstractmethod

class Sampler(object):
    """Base class for all Samplers. __iter__ is needed no matter whether you use IterableSampler
       or Squential sampler, if you want implement your own sampler, make clear what the type is
       your Dataset, if IterableDataset(method __iter__ implemented), try to use IterableSampler,
       else if you have an IndexDataset(method __getitem__ implemented), your dataset should have
       method __len__ implemented.

    """

    def __init__(self, data_source):
        pass

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class IterableSampler(Sampler):
    """Interally samples elements, used for datasets retrieved element by interator.
       yield None to act as a placeholder for each iteration

    Args:
        dataset (Dataset): set to None
    """

    def __init__(self, dataset):
        super(IterableSampler, self).__init__(None)
        self.whole_dataset = dataset

    def __iter__(self):
        while True:
            yield None

    def __len__(self):
        raise NotImplementedError("'__len__' for IterableDataset object has not defined")

class SequentialSampler(Sampler):
    """Sequentially samples elements, used for datasets retrieved element by index.

    Args:
        dataset (Dataset): index dataset(implement method __len__) for sampling
    """

    def __init__(self, dataset, distributed):
        self.whole_dataset = dataset
        self.distributed = distributed

    def __iter__(self):
        self.process_rank = 0 # The default rank is 0, which represents the main process
        self.process_size = 1 # By default, process_size=1, only the main process is running
        if self.distributed:
            import horovod.tensorflow as hvd
            hvd.init()
            self.process_rank = hvd.rank()
            self.process_size = hvd.size()
            if self.process_size < 2:
                raise EnvironmentError("The program is now trying to traverse" \
                    " the distributed TensorFlow DefaultDataLoader in only one process." \
                    " If you do not want to use distributed DataLoader, please set" \
                    " 'distributed: False'. Or If you want to use distributed DataLoader," \
                    " please set 'distributed: True' and launch multiple processes.")
        return iter(range(self.process_rank, len(self.whole_dataset), self.process_size))

    def __len__(self):
        return len(self.whole_dataset)

class BatchSampler(Sampler):
    """yield a mini-batch of indices for SquentialSampler and batch size length of None list for
       IterableSampler.

    Args:
        sampler (Sampler): sampler used for generating batches.
        batch_size (int): Size of mini-batch.
        drop_last (bool): BatchSampler will drop the last batch if drop_last is True, else
                          will return the last batch whose size will be less than batch_size

    """

    def __init__(self, sampler, batch_size, drop_last=True):
        if isinstance(drop_last, bool):
            self.drop_last = drop_last
        else:
            raise ValueError("last_batch only support bool as input")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
