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
from neural_compressor.utils.create_obj_from_config import get_algorithm

registry_algorithms = {}

def algorithm_registry(algorithm_type):
    """The class decorator used to register all Algorithm subclasses.

    Args:
        cls (class): The class of register.
        algorithm_type (str): The algorithm registration name

    Returns:
        cls: The class of register.
    """
    def decorator_algorithm(cls):
        if algorithm_type in registry_algorithms:
            raise ValueError('Cannot have two algorithms with the same name')
        registry_algorithms[algorithm_type] = cls
        return cls
    return decorator_algorithm

class ALGORITHMS(object):
    algorithms = registry_algorithms

    def __getitem__(self, algorithm_type):
        assert algorithm_type in self.algorithms, "algorithm type only support {}".\
            format(self.algorithms.keys())
        return self.algorithms[algorithm_type]

    @classmethod
    def support_algorithms(self):
        return set(self.algorithms.keys())

class AlgorithmScheduler(object):
    """control the Algorithm in different phase

    """
    def __init__(self, conf):
        self.algorithms = get_algorithm(ALGORITHMS, conf)
        self._origin_model = None
        self._q_model = None
        self._dataloader = None
        self._adaptor = None
        self._calib_iter = None

    def __call__(self):
        assert self._q_model, 'set q_model for algorithm'
        if len(self.algorithms) == 0:
            return self._q_model
        assert self._origin_model, 'set origin model for algorithm'
        assert self._dataloader, 'set dataloader for algorithm'
        assert self._adaptor, 'set adaptor for algorithm'
        assert self._calib_iter, 'set calibration iteration for algorithm'
        for algo in self.algorithms:
            self._q_model = algo(self._origin_model, 
                                 self._q_model, \
                                 self._adaptor, \
                                 self._dataloader, \
                                 self._calib_iter)
        return self._q_model

    @property
    def origin_model(self):
        return self._origin_model

    @origin_model.setter
    def origin_model(self, model):
        self._origin_model = model

    @property
    def q_model(self):
        return self._q_model

    @q_model.setter
    def q_model(self, model):
        self._q_model = model

    @property
    def dataloader(self):
        return self._dataloader

    @dataloader.setter
    def dataloader(self, dataloader):
        self._dataloader = dataloader

    @property
    def adaptor(self):
        return self._adaptor

    @adaptor.setter
    def adaptor(self, adaptor):
        self._adaptor = adaptor

    @property
    def calib_iter(self):
        return self._calib_iter

    @calib_iter.setter
    def calib_iter(self, calib_iter):
        self._calib_iter = calib_iter

class Algorithm(object):
    """ The base class of algorithm. 

    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

