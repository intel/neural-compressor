# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import logging
import time
from abc import abstractmethod
from functools import reduce
from typing import Callable, Dict

import numpy as np
import tensorflow as tf
from pkg_resources import parse_version

# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}


def version1_gte_version2(version1, version2):
    """Check if version1 is greater than or equal to version2."""
    return parse_version(version1) > parse_version(version2) or parse_version(version1) == parse_version(version2)


def register_algo(name):
    """Decorator function to register algorithms in the algos_mapping dictionary.

    Usage example:
        @register_algo(name=example_algo)
        def example_algo(model: torch.nn.Module, quant_config: RTNWeightQuantConfig) -> torch.nn.Module:
            ...
    Args:
        name (str): The name under which the algorithm function will be registered.
    Returns:
        decorator: The decorator function to be used with algorithm functions.
    """

    def decorator(algo_func):
        algos_mapping[name] = algo_func
        return algo_func

    return decorator


def deep_get(dictionary, keys, default=None):
    """Get the dot key's item in nested dict
       eg person = {'person':{'name':{'first':'John'}}}
       deep_get(person, "person.name.first") will output 'John'.

    Args:
        dictionary (dict): The dict object to get keys
        keys (dict): The deep keys
        default (object): The return item if key not exists
    Returns:
        item: the item of the deep dot keys
    """
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)


def dump_elapsed_time(customized_msg=""):
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): The parameter passed to decorator. Defaults to None.
    """

    def f(func):
        def fi(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            logging.getLogger("neural_compressor").info(
                "%s elapsed time: %s ms"
                % (customized_msg if customized_msg else func.__qualname__, round((end - start) * 1000, 2))
            )
            return res

        return fi

    return f


def default_collate(batch):  # pragma: no cover
    """Merge data with outer dimension batch size."""
    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        batch = zip(*batch)
        return [default_collate(samples) for samples in batch]
    elif isinstance(elem, np.ndarray):
        try:
            return np.stack(batch)
        except:
            return batch
    else:
        return batch


class BaseDataLoader:  # pragma: no cover
    """Base class for all DataLoaders.

    _generate_dataloader is needed to create a dataloader object
    from the general params like batch_size and sampler. The dynamic batching is just to
    generate a new dataloader by setting batch_size and last_batch.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        last_batch="rollover",
        collate_fn=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        distributed=False,
    ):
        """Initialize BaseDataLoader.

        Args:
            dataset (object): dataset from which to load the data
            batch_size (int, optional): number of samples per batch. Defaults to 1.
            last_batch (str, optional): whether to drop the last batch if it is incomplete.
                                        Support ['rollover', 'discard'], rollover means False, discard means True.
                                        Defaults to 'rollover'.
            collate_fn (callable, optional): merge data with outer dimension batch size. Defaults to None.
            sampler (Sampler, optional): Sampler object to sample data. Defaults to None.
            batch_sampler (BatchSampler, optional): BatchSampler object to generate batch of indices. Defaults to None.
            num_workers (int, optional): number of subprocesses to use for data loading. Defaults to 0.
            pin_memory (bool, optional): whether to copy data into pinned memory before returning. Defaults to False.
            shuffle (bool, optional): whether to shuffle data. Defaults to False.
            distributed (bool, optional): whether the dataloader is distributed. Defaults to False.
        """
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.last_batch = last_batch
        self.drop_last = False if last_batch == "rollover" else True

        self.dataloader = self._generate_dataloader(
            self.dataset,
            batch_size=batch_size,
            last_batch=last_batch,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            distributed=distributed,
        )

    def batch(self, batch_size, last_batch=None):
        """Set batch size for dataloader.

        Args:
            batch_size (int): number of samples per batch.
            last_batch (str, optional): whether to drop the last batch if it is incomplete.
                                        Support ['rollover', 'discard'], rollover means False, discard means True.
                                        Defaults to None.
        """
        self._batch_size = batch_size
        if last_batch is not None:
            self.last_batch = last_batch
        self.dataloader = self._generate_dataloader(
            self.dataset,
            batch_size,
            self.last_batch,
            self.collate_fn,
            self.sampler,
            self.batch_sampler,
            self.num_workers,
            self.pin_memory,
            self.shuffle,
            self.distributed,
        )

    @property
    def batch_size(self):
        """Get dataloader's batch_size.

        Returns:
            int: batch_size
        """
        return self._batch_size

    def __iter__(self):
        """Yield data in iterative order.

        Returns:
            iterator: iterator for dataloder
        """
        return iter(self.dataloader)

    @abstractmethod
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
        raise NotImplementedError


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
