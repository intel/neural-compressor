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
import random

SEARCHSPACE = {}


class SearchSpace:
    """Abstract class, the main entrance for search space.
    Using factory pattern to get the actual used class.

    Args:
        bound: A tuple or list that limit the max and min number of search space.
        interval: Only for discrete search space. Intervals in discrete space.
        value: Only for discrete search space. A list that store all the number for search.
        type: discrete or continues.

    Example:
        from neural_compressor.compression.hpo import SearchSpace
         search_space = {
        'learning_rate': SearchSpace((0.0001, 0.001)),
        'num_train_epochs': SearchSpace(bound=(20, 100), interval=1),
        'weight_decay': SearchSpace((0.0001, 0.001), type='continuous')
        }
    """

    def __new__(cls, bound=None, interval=None, value=None, type=None):
        if type is None:
            if interval is not None or value is not None:
                type = "discrete"
            else:
                type = "continuous"
        assert type in SEARCHSPACE.keys(), f"only support {list(SEARCHSPACE.keys())}"
        return SEARCHSPACE[type](bound, interval, value, type)


def register_searchspace(name):
    """Class decorator to register a SearchSpace subclass to the registry.

    Args:
        cls (class): The subclass of register.
        name: A string. Define the pruner type.

    Returns:
        cls: The class of register.
    """

    def register(search_space):
        SEARCHSPACE[name] = search_space
        return search_space

    return register


class BaseSearchSpace(object):
    """Base class for Search Space."""

    def __init__(self, bound=None, interval=None, value=None, type=None):
        """Initialize."""
        if bound:
            if not isinstance(bound, (list, tuple)):  # pragma: no cover
                raise TypeError("bound should be list or tuple, not {}".format(type(bound)))
            if len(bound) != 2:  # pragma: no cover
                raise ValueError("bound should only contain two elements, [start, end)")
            if bound[1] <= bound[0]:  # pragma: no cover
                raise ValueError("empty range for [{}, {})".format(bound[0], bound[1]))
        assert value or bound, "must set value or bound to initialize the search space"

        self.bound = bound
        self.interval = interval
        self.value = value
        self.type = type
        if type == "discrete":
            if value:
                self.total_num = len(value)
            else:
                self.total_num = int((bound[1] - bound[0]) / interval)
        else:
            self.total_num = float("inf")

    def get_value(self):
        """Get one value from the search space."""
        pass


@register_searchspace("discrete")
class DiscreteSearchSpace(BaseSearchSpace):
    """Discrete Search Space."""

    def __init__(self, bound=None, interval=None, value=None, type=None):
        if bound and interval is None:
            if isinstance(bound[0], int) and isinstance(bound[1], int):
                interval = 1
            else:
                interval = 0.01
        super().__init__(bound=bound, interval=interval, value=value, type="discrete")

    def get_random_value(self):
        """Get a random value from search space."""
        idx = random.randint(0, self.total_num - 1)
        return self.get_nth_value(idx)

    def get_nth_value(self, idx):
        """Get the number n value from search space."""
        if self.bound:
            return round(self.bound[0] + idx * self.interval, 10)
        else:
            return self.value[idx]

    def get_all(self):
        """Get all values from search space."""
        return [self.get_nth_value(i) for i in range(self.total_num)]

    def get_value(self, idx=None):
        """Get number n value from search space if idx is given.

        Otherwise, get a random value.
        """
        if idx is not None:
            if not isinstance(idx, int):
                raise TypeError("The type of idx should be int, not {}".format(type(idx)))
            if idx < 0:
                return self.get_all()
            value = self.get_nth_value(idx)
        else:
            value = self.get_random_value()
        return value

    def index(self, value):
        """Return the index of the value."""
        if self.value:
            return self.value.index(value)
        else:
            return int((value - self.bound[0]) / self.interval)


@register_searchspace("continuous")
class ContinuousSearchSpace(BaseSearchSpace):
    """Continuous Search Space."""

    def __init__(self, bound, interval=None, value=None, type=None):
        super().__init__(bound, interval, value, "continuous")

    def get_value(self):
        """Get one value from the search space."""
        if self.bound[1] > 1:
            int_num = random.randrange(int(self.bound[0]), int(self.bound[1]) + 1)
        else:
            int_num = 0
        while True:
            value = random.random() * self.bound[1]
            value = int_num + value
            if value > self.bound[0] and value < self.bound[1]:
                break
        return value
