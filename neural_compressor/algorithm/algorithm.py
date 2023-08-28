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
"""Register algorithms."""

from abc import abstractmethod

from neural_compressor.utils.create_obj_from_config import get_algorithm

# {location: {algorithm_type: cls}}
registry_algorithms = {}


def algorithm_registry(algorithm_type, location):
    """Decorate and register all Algorithm subclasses.

    Args:
        cls (class): The class of register.
        algorithm_type (str): The algorithm registration name
        location (str): The location to call algorithms

    Returns:
        cls: The class of register.
    """

    def decorator_algorithm(cls):
        if location in registry_algorithms and algorithm_type in registry_algorithms[location]:
            raise ValueError("Cannot have two algorithms with the same name")

        if location not in registry_algorithms:
            registry_algorithms[location] = {}
        registry_algorithms[location][algorithm_type] = cls()
        return cls

    return decorator_algorithm


class ALGORITHMS(object):
    """Build a dict for registered algorithms."""

    algorithms = registry_algorithms

    def __getitem__(self, algorithm_type):
        """Return algorithm class by algorithm_type name.

        Args:
            algorithm_type (str):  The algorithm registration name

        Returns:
            cls (class): The class of algorithm.
        """
        result = None
        for location in self.algorithms:
            for key in self.algorithms[location]:
                if key == algorithm_type:
                    result = self.algorithms[location][key]
        assert result, "algorithm type only support {}".format(self.support_algorithms())
        return result

    @classmethod
    def support_algorithms(self):
        """Get all algorithms.

        Returns:
            Set: A set of all algorithms.
        """
        supported_algos = set([self.algorithms[key] for key in self.algorithms])
        return supported_algos


class AlgorithmScheduler(object):
    """Control the Algorithm in different phase."""

    def __init__(self, conf):
        """Initialize AlgorithmScheduler.

        Args:
            conf (dict): Configuration of algorithm.
        """
        self._exec_algorithms = {}
        self._origin_model = None
        self._q_model = None
        self._dataloader = None
        self._adaptor = None
        self._calib_iter = None

    def append_algorithm(self, location, algorithm):
        """Append algorithm to list of executed algorithms.

        Args:
            location: The location to call algorithm
            algorithm: algorithm instance
        """
        self._exec_algorithms[location] = self._exec_algorithms.get(location, [])
        self._exec_algorithms[location].append(algorithm)

    def reset_exec_algorithms(self):
        """Reset the list of executed algorithms."""
        self._exec_algorithms = {}

    def __call__(self, location):
        """Return the processed model via algorithm.

        Returns:
            model: The framework model.
        """
        assert self._q_model, "set q_model for algorithm"
        if len(self._exec_algorithms.get(location, [])) == 0:
            return self._q_model
        assert self._origin_model, "set origin model for algorithm"
        assert self._adaptor, "set adaptor for algorithm"
        assert self._calib_iter, "set calibration iteration for algorithm"
        for algo in self._exec_algorithms.get(location, []):
            self._q_model = algo(self._origin_model, self._q_model, self._adaptor, self._dataloader, self._calib_iter)
        return self._q_model

    @property
    def origin_model(self):
        """Return the origin model.

        Returns:
            model: The origin model.
        """
        return self._origin_model

    @origin_model.setter
    def origin_model(self, model):
        """Set the origin model.

        Args:
            model: The origin model.
        """
        self._origin_model = model

    @property
    def q_model(self):
        """Return the quantized model.

        Returns:
            model: The quantized model.
        """
        return self._q_model

    @q_model.setter
    def q_model(self, model):
        """Set the quantized model.

        Args:
            model: The quantized model.
        """
        self._q_model = model

    @property
    def dataloader(self):
        """Return the dataloader.

        Returns:
            dataloader: The dataloader.
        """
        return self._dataloader

    @dataloader.setter
    def dataloader(self, dataloader):
        """Set the dataloader.

        Args:
            dataloader: The dataloader.
        """
        self._dataloader = dataloader

    @property
    def adaptor(self):
        """Return the adaptor.

        Returns:
            adaptor: The adaptor.
        """
        return self._adaptor

    @adaptor.setter
    def adaptor(self, adaptor):
        """Set the adaptor.

        Args:
            adaptor: The adaptor.
        """
        self._adaptor = adaptor

    @property
    def calib_iter(self):
        """Return the calibration iter number.

        Returns:
            calib_iter: The calibration iter number.
        """
        return self._calib_iter

    @calib_iter.setter
    def calib_iter(self, calib_iter):
        """Set the calibration iter number.

        Args:
            calib_iter: The calibration iter number
        """
        self._calib_iter = calib_iter


class Algorithm(object):
    """The base class of algorithm."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Raise NotImplementedError.

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError
