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
import numpy as np

registry_patterns = {}

def pattern_registry(pattern_type):
    """The class decorator used to register all Pruning Pattern subclasses.

    Args:
        cls (class): The class of register.
        pattern_type (str): The pattern registration name

    Returns:
        cls: The class of register.
    """
    def decorator_pattern(cls):
        if pattern_type in registry_patterns:
            raise ValueError('Cannot have two pattern with the same name')
        registry_patterns[pattern_type] = cls
        return cls
    return decorator_pattern

class PATTERNS(object):
    patterns = registry_patterns

    def __getitem__(self, pattern_type):
        assert pattern_type in self.patterns, "pattern type only support {}".\
            format(self.patterns.keys())
        return self.patterns[pattern_type]

    @classmethod
    def support_pattern(self):
        return set(self.patterns.keys())

class PatternBase:
    """ base class of pruning pattern """
    def __init__(self, mask_shape, is_contiguous=True):
        self.mask_shape = mask_shape
        self.is_contiguous = is_contiguous

    def compute_sparsity(self, tensor):
        raise NotImplementedError

    def reduce(self, tensor, method='abs_sum'):
        """ reshaped tensor, support 'abs_max', 'abs_sum' """
        assert tensor.shape[-1] % self.mask_shape[-1] == 0 and \
                tensor.shape[-2] % self.mask_shape[-2] == 0, \
            'tensor shape {} cannot be divided by mask {}'.format(tensor.shape, self.mask_shape)

        dims = list(range(len(tensor.shape) + 2))
        new_shape = list(tensor.shape)[:-2]
        new_shape.append(tensor.shape[-2] // self.mask_shape[-2])
        new_shape.append(self.mask_shape[-2])
        new_shape.append(tensor.shape[-1] // self.mask_shape[-1])
        new_shape.append(self.mask_shape[-1])

        reshaped_tensor = tensor.reshape(new_shape)
        new_tensor = np.transpose(reshaped_tensor, dims[:-3] + [dims[-2], dims[-3], dims[-1]])
        reduced_tensor = new_tensor.reshape(new_shape[:-3] + [new_shape[-2], -1])
        if method == 'abs_max':
            return np.abs(reduced_tensor).max(-1).values
        elif method == 'abs_sum':
            return np.abs(reduced_tensor).sum(-1)
        else:
            raise NotImplementedError
