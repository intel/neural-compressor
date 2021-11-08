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

from collections import OrderedDict
import numpy as np


class Tensor(object):
    def __init__(self,
                 name='',
                 source_op=[],
                 dest_op=[],
                 shape=None,
                 data=None,
                 dtype=None,
                 location=None):
        self._name = name
        # assume data in neural_compressor tensor should be numpy array
        # however, we don't assign the data diretly if the tensor is
        # const like weight when parse model
        # otherwise it will make a bloated new graph
        # but it still can be set when using the constructed new graph
        self._data = data
        self._shape = shape
        self._dtype = dtype
        # location in bin file if const
        self._location = location
        self._source_op = source_op
        self._dest_op = dest_op

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        self._location = location

    @property
    def source_op(self):
        return self._source_op

    @source_op.setter
    def source_op(self, source_op):
        self._source_op = source_op

    @property
    def dest_op(self):
        return self._dest_op

    @dest_op.setter
    def dest_op(self, dest_op):
        self._dest_op = dest_op

    @property
    def config(self):
        conf_dict = OrderedDict()
        if self._dtype is not None:
            conf_dict['dtype'] = self._dtype
        if self._shape is not None:
            conf_dict['shape'] = self._shape
        if self._location is not None:
            conf_dict['location'] = self._location

        return conf_dict
