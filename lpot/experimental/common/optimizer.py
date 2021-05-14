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
class TensorflowOptimizers(object):
    def __init__(self):
        self.optimizers = {}
        self.optimizers.update(TENSORFLOW_OPTIMIZERS)

@singleton
class PyTorchOptimizers(object):
    def __init__(self):
        self.optimizers = {}
        self.optimizers.update(PYTORCH_OPTIMIZERS)

framework_optimizers = {"tensorflow": TensorflowOptimizers,
                        "pytorch":    PyTorchOptimizers}

# user/model specific optimizers will be registered here
TENSORFLOW_OPTIMIZERS = {}
PYTORCH_OPTIMIZERS= {}

registry_optimizers = {"tensorflow": TENSORFLOW_OPTIMIZERS,
                       "pytorch":    PYTORCH_OPTIMIZERS}

class Optimizers(object):
    def __init__(self, framework):
        assert framework in ("tensorflow", "pytorch"), \
                             "framework support tensorflow pytorch"
        self.optimizers = framework_optimizers[framework]().optimizers

    def __getitem__(self, optimizer_type):
        assert optimizer_type in self.optimizers.keys(), "only support optimizers in {}".\
            format(self.optimizers.keys())

        return self.optimizers[optimizer_type]

    def register(self, name, optimizer_cls):
        assert name not in self.optimizers.keys(), 'registered optimizer name already exists.'
        self.optimizers.update({name: optimizer_cls})

def optimizer_registry(optimizer_type, framework):
    """The class decorator used to register all Optimizer subclasses.
       cross framework optimizer is supported by add param as framework='tensorflow, pytorch'

    Args:
        optimizer_type (str): The string of supported criterion.
        framework (str): The string of supported framework.

    Returns:
        cls: The class of register.
    """
    def decorator_optimizer(cls):
        for fw in [fwk.strip() for fwk in framework.split(',')]:
            assert fw in [
                "tensorflow",
                "pytorch"], "The framework support tensorflow pytorch"

            if optimizer_type in registry_optimizers[fw ].keys():
                raise ValueError('Cannot have two optimizers with the same name')
            registry_optimizers[fw][optimizer_type] = cls
        return cls
    return decorator_optimizer

@optimizer_registry('SGD', 'tensorflow')
class TensorFlowSGD(object):
    """TensorFlow keras SGD optimizer.

    Args:
        param_dict (dict): The dict of parameters setting by user for SGD optimizer
    """
    def __init__(self, param_dict):
        assert isinstance(param_dict, dict), 'This optimizer constructor parameter must be a dict'
        self._param_dict = param_dict

    def _mapping(self):
        _param_map = {'learning_rate': 'learning_rate',
                      'momentum': 'momentum',
                      'nesterov': 'nesterov'}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                _dict.update({_param_map[key] : self._param_dict[key]})
        return _dict

    def __call__(self, **kwargs):
        return tf.keras.optimizers.SGD, self._mapping(**kwargs)

@optimizer_registry('SGD', 'pytorch')
class PyTorchSGD(object):
    """PyTorch SGD optimizer.

    Args:
        param_dict (dict): The dict of parameters setting by user for SGD optimizer
    """
    def __init__(self, param_dict):
        assert isinstance(param_dict, dict), 'This optimizer constructor parameter must be a dict'
        self._param_dict = param_dict

    def _mapping(self):
        _param_map = {'learning_rate': 'lr',
                      'momentum': 'momentum',
                      'nesterov': 'nesterov',
                      'weight_decay': 'weight_decay'}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                _dict.update({_param_map[key] : self._param_dict[key]})
        return _dict

    def __call__(self, **kwargs):
        return torch.optim.SGD, self._mapping(**kwargs)
