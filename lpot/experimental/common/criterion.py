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
from lpot.utils.utility import LazyImport, singleton

torch = LazyImport('torch')
tf = LazyImport('tensorflow')

@singleton
class TensorflowCriterions(object):
    def __init__(self):
        self.criterions = {}
        self.criterions.update(TENSORFLOW_CRITERIONS)

@singleton
class PyTorchCriterions(object):
    def __init__(self):
        self.criterions = {}
        self.criterions.update(PYTORCH_CRITERIONS)

framework_criterions = {"tensorflow": TensorflowCriterions,
                        "pytorch":    PyTorchCriterions}

# user/model specific criterions will be registered here
TENSORFLOW_CRITERIONS = {}
PYTORCH_CRITERIONS= {}

registry_criterions = {"tensorflow": TENSORFLOW_CRITERIONS,
                       "pytorch":    PYTORCH_CRITERIONS}

class Criterions(object):
    def __init__(self, framework):
        assert framework in ("tensorflow", "pytorch"), \
                             "framework support tensorflow pytorch"
        self.criterions = framework_criterions[framework]().criterions

    def __getitem__(self, criterion_type):
        assert criterion_type in self.criterions.keys(), "only support criterions in {}".\
            format(self.criterions.keys())

        return self.criterions[criterion_type]

    def register(self, name, criterion_cls):
        assert name not in self.criterions.keys(), 'registered criterion name already exists.'
        self.criterions.update({name: criterion_cls})

def criterion_registry(criterion_type, framework):
    """The class decorator used to register all criterion classes.

    Args:
        criterion_type (str): The string of supported criterion.
        framework (str): The string of supported framework. 

    Returns:
        cls: The class of register.
    """
    def decorator_criterion(cls):
        for fw in [fwk.strip() for fwk in framework.split(',')]:
            assert fw in [
                "tensorflow",
                "pytorch"], "The framework support tensorflow pytorch"

            if criterion_type in registry_criterions[fw].keys():
                raise ValueError('Cannot have two criterions with the same name')
            registry_criterions[fw][criterion_type] = cls
        return cls
    return decorator_criterion

@criterion_registry('CrossEntropyLoss', 'tensorflow')
class TensorFlowCrossEntropyLoss(object):
    """TensorFlow CrossEntropyLoss criterion.

    Args:
        param_dict (dict): The dict of parameters setting by user for CrossEntropyLoss criterion.
    """
    def __init__(self, param_dict):
        assert isinstance(param_dict, dict), 'This criterion constructor parameter must be a dict'
        self._param_dict = param_dict

    def _mapping(self):
        _param_map = {'reduction': 'reduction'}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                if key == 'reduction':
                    assert self._param_dict[key] in ['none', 'mean', 'sum'], \
                        'Supported reduction value is none, mean, sum'
                _dict.update({_param_map[key]: self._param_dict[key]})
        return _dict

    def __call__(self, **kwargs):
        return tf.keras.losses.CategoricalCrossentropy, self._mapping(**kwargs)

@criterion_registry('CrossEntropyLoss', 'pytorch')
class PyTorchCrossEntropyLoss(object):
    """PyTorch CrossEntropyLoss criterion.

    Args:
        param_dict (dict): The dict of parameters setting by user for SGD criterion
    """
    def __init__(self, param_dict):
        assert isinstance(param_dict, dict), 'This criterion constructor parameter must be a dict'
        self._param_dict = param_dict

    def _mapping(self):
        _param_map = {'reduction': 'reduction'}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                if key == 'reduction':
                    assert self._param_dict[key] in ['none', 'mean', 'sum'], \
                        'Supported reduction value is none, mean, sum'
                _dict.update({_param_map[key]: self._param_dict[key]})
        return _dict

    def __call__(self, **kwargs):
        return torch.nn.CrossEntropyLoss, self._mapping(**kwargs)
