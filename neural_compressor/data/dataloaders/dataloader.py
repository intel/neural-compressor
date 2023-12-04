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
"""Built-in dataloaders for multiple framework backends."""
from .mxnet_dataloader import MXNetDataLoader
from .onnxrt_dataloader import ONNXRTDataLoader
from .pytorch_dataloader import PyTorchDataLoader
from .tensorflow_dataloader import TensorflowDataLoader

DATALOADERS = {
    "tensorflow": TensorflowDataLoader,
    "tensorflow_itex": TensorflowDataLoader,
    "keras": TensorflowDataLoader,
    "mxnet": MXNetDataLoader,
    "pytorch": PyTorchDataLoader,
    "pytorch_ipex": PyTorchDataLoader,
    "pytorch_fx": PyTorchDataLoader,
    "onnxruntime": ONNXRTDataLoader,
    "onnxrt_qlinearops": ONNXRTDataLoader,
    "onnxrt_integerops": ONNXRTDataLoader,
    "onnxrt_qdq": ONNXRTDataLoader,
    "onnxrt_qoperator": ONNXRTDataLoader,
}


class DataLoader(object):
    """Entrance of all configured DataLoaders."""

    def __new__(
        cls,
        framework,
        dataset,
        batch_size=1,
        collate_fn=None,
        last_batch="rollover",
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        distributed=False,
    ):
        """Initialize a Dataloader with needed information.

        Args:
            framework (str): different frameworks, such as tensorflow, pytorch, onnx.
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
        assert framework in (
            "tensorflow",
            "tensorflow_itex",
            "keras",
            "pytorch",
            "pytorch_ipex",
            "pytorch_fx",
            "onnxruntime",
            "onnxrt_qdqops",
            "onnxrt_qlinearops",
            "onnxrt_integerops",
            "mxnet",
        ), "framework support tensorflow pytorch mxnet onnxruntime"
        return DATALOADERS[framework](
            dataset=dataset,
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
        assert hasattr(dataloader, "__iter__") and hasattr(
            dataloader, "batch_size"
        ), "dataloader must implement __iter__ method and batch_size attribute"
        assert (
            not distributed
        ), "Please use \
            neural_compressor.data.DataLoader to support distributed computing"
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
            distributed=bool(dataloader.distributed or distributed),
        )


def check_dataloader(dataloader):
    """Check if the dataloader meets requirement of neural_compressor."""
    assert hasattr(dataloader, "__iter__") and hasattr(
        dataloader, "batch_size"
    ), "dataloader must implement __iter__ method and batch_size attribute"
    return True
