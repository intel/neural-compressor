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
"""TensorFlow Dataloader implementation."""

import collections
import logging
import sys
from abc import abstractmethod
from math import ceil, floor

import numpy as np

from neural_compressor.utils.utility import LazyImport

from ..datasets.bert_dataset import TensorflowBertDataset, TensorflowModelZooBertDataset
from .base_dataloader import BaseDataLoader
from .default_dataloader import DefaultDataLoader, default_collate
from .fetcher import FETCHERS
from .sampler import BatchSampler, IterableSampler, SequentialSampler

tf = LazyImport("tensorflow")
neural_compressor = LazyImport("neural_compressor")


class TFDataDataLoader(BaseDataLoader):  # pragma: no cover
    """Tensorflow dataloader class.

    In tensorflow1.x dataloader is coupled with the graph, but it also support feed_dict
    method to do session run, this dataloader is designed to satisfy the usage of feed dict
    in tf1.x. Although it's a general dataloader and can be used in MXNet and PyTorch.

    Args:
        dataset: obj. wrapper of needed data.
        batch_size: int. batch size
    """

    def __init__(self, dataset, batch_size=1, last_batch="rollover"):
        """Initialize `TFDataDataLoader` class."""
        self.dataset = dataset
        self.last_batch = last_batch
        self._batch_size = batch_size
        dataset = dataset.batch(batch_size)

    def batch(self, batch_size, last_batch="rollover"):
        """Dataset return data per batch."""
        drop_last = False if last_batch == "rollover" else True
        self._batch_size = batch_size
        self.dataset = self.dataset.batch(batch_size, drop_last)

    def __iter__(self):
        """Iterate dataloader."""
        return self._generate_dataloader(
            self.dataset,
            batch_size=self.batch_size,
            last_batch=self.last_batch,
        )

    def _generate_dataloader(
        self,
        dataset,
        batch_size=1,
        last_batch="rollover",
        collate_fn=None,
        sampler=None,
        batch_sampler=None,
        num_workers=None,
        pin_memory=None,
        shuffle=False,
        distributed=False,
    ):
        """Yield data."""
        drop_last = False if last_batch == "rollover" else True
        if shuffle:
            logging.warning("Shuffle is not supported yet in TFDataLoader, " "ignoring shuffle keyword.")

        def check_dynamic_shape(element_spec):
            if isinstance(element_spec, collections.abc.Sequence):
                return any([check_dynamic_shape(ele) for ele in element_spec])
            elif isinstance(element_spec, tf.TensorSpec):
                return True if element_spec.shape.num_elements() is None else False
            else:
                raise ValueError("unrecognized element spec...")

        def squeeze_output(output):
            if isinstance(output, collections.abc.Sequence):
                return [squeeze_output(ele) for ele in output]
            elif isinstance(output, np.ndarray):
                return np.squeeze(output, axis=0)
            else:
                raise ValueError("not supported output format....")

        if tf.executing_eagerly():
            index = 0
            outputs = []
            for iter_tensors in dataset:
                samples = []
                iter_inputs, iter_labels = iter_tensors[0], iter_tensors[1]
                if isinstance(iter_inputs, tf.Tensor):
                    samples.append(iter_inputs.numpy())
                else:
                    samples.append(tuple(iter_input.numpy() for iter_input in iter_inputs))
                if isinstance(iter_labels, tf.Tensor):
                    samples.append(iter_labels.numpy())
                else:
                    samples.append([np.array(l) for l in iter_labels])
                index += 1
                outputs.append(samples)
                if index == batch_size:
                    outputs = default_collate(outputs)
                    yield outputs
                    outputs = []
                    index = 0
            if len(outputs) > 0:
                outputs = default_collate(outputs)
                yield outputs
        else:
            try_single_batch = check_dynamic_shape(dataset.element_spec)
            dataset = dataset.batch(1 if try_single_batch else batch_size, drop_last)
            ds_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            iter_tensors = ds_iterator.get_next()
            data_config = tf.compat.v1.ConfigProto()
            data_config.use_per_session_threads = 1
            data_config.intra_op_parallelism_threads = 1
            data_config.inter_op_parallelism_threads = 16
            data_sess = tf.compat.v1.Session(config=data_config)
            # pylint: disable=no-name-in-module
            from tensorflow.python.framework.errors_impl import OutOfRangeError

            while True:
                if not try_single_batch:
                    try:
                        outputs = data_sess.run(iter_tensors)
                        yield outputs
                    except OutOfRangeError:
                        data_sess.close()
                        return
                else:
                    try:
                        outputs = []
                        for i in range(0, batch_size):
                            outputs.append(squeeze_output(data_sess.run(iter_tensors)))
                        outputs = default_collate(outputs)
                        yield outputs
                    except OutOfRangeError:
                        if len(outputs) == 0:
                            data_sess.close()
                            return
                        else:
                            outputs = default_collate(outputs)
                            yield outputs
                            data_sess.close()
                            return


class TensorflowBertDataLoader(DefaultDataLoader):  # pragma: no cover
    """Subclass of DefaultDataLoader.

    this dataloader is designed to satisfy the usage of Bert models.
    """

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
        if shuffle:
            logging.warning("Shuffle is not supported yet in TensorflowBertDataLoader, " "ignoring shuffle keyword.")

        def bert_collate_fn(batch):
            elem = batch[0]
            return elem

        drop_last = False if last_batch == "rollover" else True
        sampler = self._generate_sampler(dataset, distributed)
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.fetcher = FETCHERS[self.dataset_type](dataset, bert_collate_fn, drop_last, distributed)

        for batched_indices in self.batch_sampler:
            try:
                data = self.fetcher(batched_indices)
                yield (data[0], batch_size), data[1]
            except StopIteration:
                return


class TensorflowModelZooBertDataLoader(DefaultDataLoader):  # pragma: no cover
    """Subclass of DefaultDataLoader.

    this dataloader is designed to satisfy the usage of Model Zoo Bert models.
    """

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
        if shuffle:
            logging.warning("Shuffle is not supported yet in TensorflowBertDataLoader, " "ignoring shuffle keyword.")

        def bert_collate_fn(batch):
            input_ids = []
            input_mask = []
            segment_ids = []
            for elem in batch:
                input_ids.append(elem[0][0][0])
                input_mask.append(elem[0][1][0])
                segment_ids.append(elem[0][2][0])
            inputs = [input_ids, input_mask, segment_ids]
            return inputs, batch[0][1]

        drop_last = False if last_batch == "rollover" else True
        sampler = self._generate_sampler(dataset, distributed)
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.fetcher = FETCHERS[self.dataset_type](dataset, bert_collate_fn, drop_last, distributed)

        inputs = []
        for batched_indices in self.batch_sampler:
            try:
                data = self.fetcher(batched_indices)
                yield data
            except StopIteration:
                return


class TensorflowDataLoader(BaseDataLoader):  # pragma: no cover
    """DataLoader for framework Tensorflow.

    if it's a tf.data.Dataset we will directly use the dataloader in the other case
    will use DefaultDataLoader instead.
    """

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
        if shuffle:
            logging.warning("Shuffle is not supported yet in TensorflowDataLoader, " "ignoring shuffle keyword.")
        if isinstance(dataset, tf.data.Dataset):
            if int(tf.__version__[0]) > 1:
                has_batch = hasattr(dataset, "_batch_size")
            else:
                has_batch = hasattr(dataset._dataset, "_batch_size")
            if has_batch:
                raise TypeError(
                    f"Parameter 'batch_size={batch_size}'"
                    " conflicts with 'tf.data.Dataset',"
                    f" because {dataset} is already a BatchDataset."
                    f" Please pass in 'tf.data.Dataset' without batch attributes."
                )
            process_rank = 0  # The default rank is 0, which represents the main process
            process_size = 1  # By default, process_size=1, only the main process is running
            if self.distributed:
                import horovod.tensorflow as hvd

                hvd.init()
                process_rank = hvd.rank()
                process_size = hvd.size()
                if process_size < 2:
                    raise EnvironmentError(
                        "The program is now trying to generate"
                        " the distributed TensorflowDataLoader in only one process."
                        " If you do not want to use distributed DataLoader, please set"
                        " 'distributed: False'. Or If you want to use distributed DataLoader,"
                        " please set 'distributed: True' and launch multiple processes."
                    )
                dataset = dataset.shard(process_size, process_rank)
            tf_dataloader = TFDataDataLoader(dataset, batch_size, last_batch=last_batch)
            return tf_dataloader
        elif isinstance(dataset, TensorflowBertDataset):
            if distributed:
                raise NotImplementedError(
                    "Distributed TensorflowBertDataLoader" " is not yet supported, please set 'distributed: False'"
                )
            tf_bert_dataloader = TensorflowBertDataLoader(
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
            )
            return tf_bert_dataloader
        elif isinstance(dataset, TensorflowModelZooBertDataset):
            if distributed:
                raise NotImplementedError(
                    "Distributed TensorflowBertDataLoader" " is not yet supported, please set 'distributed: False'"
                )
            tf_bert_dataloader = TensorflowModelZooBertDataLoader(
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
            )
            return tf_bert_dataloader
        else:
            return DefaultDataLoader(
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
            )

    def __bool__(self):
        """Judgement if the dataloader exists."""
        # workaround in assert dataloader which will overload __len__() without __bool__()
        # provided. Calling __len__() in asserting is not supposed and may cause issues.
        return True

    def __len__(self):
        """Total number of dataset."""
        try:
            dataset_len = self.dataset.__len__()
        except (AttributeError, TypeError):
            try:
                dataset_len = 0
                for _ in self.dataset:
                    dataset_len += 1
            except RuntimeError:
                return sum([1 for _ in self])
        except Exception:
            raise ValueError(
                f"{self.dataset} is invalid, {self.dataset}"
                " does not support calculating the length of its dataloader"
            )
        process_rank = 0  # The default rank is 0, which represents the main process
        process_size = 1  # By default, process_size=1, only the main process is running
        if self.distributed:
            import horovod.tensorflow as hvd

            hvd.init()
            process_rank = hvd.rank()
            process_size = hvd.size()
            if process_size < 2:
                raise EnvironmentError(
                    "The program is now trying to get length of"
                    " the distributed TensorflowDataLoader in only one process."
                    " If you do not want to use distributed DataLoader, please set"
                    " 'distributed: False'. Or If you want to use distributed DataLoader,"
                    " please set 'distributed: True' and launch multiple processes."
                )
        if process_rank < (dataset_len % process_size):
            self.dis_dataset_len = dataset_len // process_size + 1
        else:
            self.dis_dataset_len = dataset_len // process_size
        if self.drop_last is False:
            dataloader_len = ceil(self.dis_dataset_len / self.batch_size)
        else:
            dataloader_len = floor(self.dis_dataset_len / self.batch_size)
        return sys.maxsize if dataloader_len > sys.maxsize else dataloader_len
