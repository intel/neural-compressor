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

from abc import abstractmethod

from lpot.utils.utility import LazyImport, singleton
torchvision = LazyImport('torchvision')
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')


@singleton
class TensorflowDatasets(object):
    def __init__(self):
        self.datasets = {
            "TFRecordDataset": tf.data.TFRecordDataset,
        }
        self.datasets.update(TENSORFLOW_DATASETS)


@singleton
class PyTorchDatasets(object):
    def __init__(self):
        self.datasets = {
            'ImageFolder': PytorchMxnetWrapDataset(
                                torchvision.datasets.ImageFolder),
            'DatasetFolder': PytorchMxnetWrapDataset(
                                torchvision.datasets.DatasetFolder),
            'ImageNet': PytorchMxnetWrapDataset(
                                torchvision.datasets.ImageNet),
        }
        self.datasets.update(PYTORCH_DATASETS)


@singleton
class MXNetDatasets(object):
    def __init__(self):
        self.datasets = {
            "ImageRecordDataset": PytorchMxnetWrapDataset(
                                    mx.gluon.data.vision.datasets.ImageRecordDataset),
            "ImageFolderDataset": PytorchMxnetWrapDataset(
                                    mx.gluon.data.vision.datasets.ImageFolderDataset),
        }
        self.datasets.update(MXNET_DATASETS)

@singleton
class ONNXRTQLDatasets(object):
    def __init__(self):
        self.datasets = {}
        self.datasets.update(ONNXRTQL_DATASETS)

@singleton
class ONNXRTITDatasets(object):
    def __init__(self):
        self.datasets = {}
        self.datasets.update(ONNXRTIT_DATASETS)



framework_datasets = {"tensorflow": TensorflowDatasets,
                      "mxnet": MXNetDatasets,
                      "pytorch": PyTorchDatasets, 
                      "pytorch_ipex": PyTorchDatasets, 
                      "onnxrt_qlinearops": ONNXRTQLDatasets,
                      "onnxrt_integerops": ONNXRTITDatasets}

"""The datasets supported by lpot, it's model specific and can be configured by yaml file.

   User could add new datasets by implementing new Dataset subclass under this directory.
   The naming convention of new dataset subclass should be something like ImageClassifier, user
   could choose this dataset by setting "imageclassifier" string in tuning.strategy field of yaml.

   DATASETS variable is used to store all implelmented Dataset subclasses to support
   model specific dataset.
"""


class DATASETS(object):
    def __init__(self, framework):
        assert framework in ["tensorflow", "mxnet", "onnxrt_qlinearops", "onnxrt_integerops",
                             "pytorch", "pytorch_ipex"], \
                             "framework support tensorflow pytorch mxnet onnxrt"
        self.datasets = framework_datasets[framework]().datasets

    def __getitem__(self, dataset_type):
        assert dataset_type in self.datasets.keys(), "dataset type only support {}".\
            format(self.datasets.keys())
        return self.datasets[dataset_type]

class PytorchMxnetWrapDataset():
    def __init__(self, datafunc):
        self.datafunc = datafunc

    def __call__(self, transform=None, filter=None, *args, **kwargs):
        return PytorchMxnetWrapFunction(self.datafunc, transform=transform, \
            filter=filter, *args, **kwargs)

class PytorchMxnetWrapFunction():
    def __init__(self, dataset, transform, filter, *args, **kwargs):
        self.dataset = dataset(*args, **kwargs)
        self.transform = transform
        self.filter = filter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

# user/model specific datasets will be registered here
TENSORFLOW_DATASETS = {}
MXNET_DATASETS = {}
PYTORCH_DATASETS = {}
PYTORCHIPEX_DATASETS = {}
ONNXRTQL_DATASETS = {}
ONNXRTIT_DATASETS = {}

registry_datasets = {"tensorflow": TENSORFLOW_DATASETS,
                     "mxnet": MXNET_DATASETS,
                     "pytorch": PYTORCH_DATASETS,
                     "pytorch_ipex": PYTORCHIPEX_DATASETS,
                     "onnxrt_integerops": ONNXRTQL_DATASETS,
                     "onnxrt_qlinearops": ONNXRTIT_DATASETS}


def dataset_registry(dataset_type, framework, dataset_format=''):
    """The class decorator used to register all Dataset subclasses.


    Args:
        cls (class): The class of register.
        dataset_type (str): The dataset registration name
        framework (str): support 3 framework including 'tensorflow', 'pytorch', 'mxnet'
        data_format (str): The format dataset saved, eg 'raw_image', 'tfrecord'

    Returns:
        cls: The class of register.
    """
    def decorator_dataset(cls):
        for single_framework in [fwk.strip() for fwk in framework.split(',')]:
            assert single_framework in [
                "tensorflow",
                "mxnet",
                "pytorch",
                "pytorch_ipex",
                "onnxrt_qlinearops",
                "onnxrt_integerops"
            ], "The framework support tensorflow mxnet pytorch onnxrt"
            dataset_name = dataset_type + dataset_format
            if dataset_name in registry_datasets[single_framework].keys():
                raise ValueError('Cannot have two datasets with the same name')
            registry_datasets[single_framework][dataset_name] = cls
        return cls
    return decorator_dataset


class Dataset(object):
    """ The base class of dataset. Subclass datasets should overwrite two methods:
    `__getitem__` for indexing to data sample and `__len__`for the size of the dataset

    """

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    # it's suggested to implement your  __len__ method though we do not set it in abstract class
    # @abstractmethod
    # def __len__(self):
    #     raise NotImplementedError


class IterableDataset(object):
    """An iterable Dataset. Subclass iterable dataset should aslo implement a method:
    `__iter__` for interating over the samples of the dataset.

    """

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError
