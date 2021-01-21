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
from lpot.utils.utility import singleton

@singleton
class TensorflowFilters(object):
    def __init__(self):
        self.filters = {}
        self.filters.update(TENSORFLOW_FILTERS)

TENSORFLOW_FILTERS = {}

framework_filters = {"tensorflow": TensorflowFilters}

registry_filters = {"tensorflow": TENSORFLOW_FILTERS}

class FILTERS(object):
    def __init__(self, framework):
        assert framework in ("tensorflow"), \
                             "framework support tensorflow pytorch mxnet onnxrt"
        self.filters = framework_filters[framework]().filters
        self.framework = framework

    def __getitem__(self, filter_type):
        assert filter_type in self.filters.keys(), "filter support {}".\
            format(self.filters.keys())
        return self.filters[filter_type]


def filter_registry(filter_type, framework):
    """The class decorator used to register all transform subclasses.


    Args:
        filter_type (str): fILTER registration name
        framework (str): support 4 framework including 'tensorflow', 'pytorch', 'mxnet', 'onnxrt'
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """
    def decorator_transform(cls):
        for single_framework in [fwk.strip() for fwk in framework.split(',')]:
            assert single_framework in [
                "tensorflow"
            ], "The framework support tensorflow"
            if filter_type in registry_filters[single_framework].keys():
                raise ValueError('Cannot have two transforms with the same name')
            registry_filters[single_framework][filter_type] = cls
        return cls
    return decorator_transform

class Filter(object):
    """The base class for transform. __call__ method is needed when write user specific transform

    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError