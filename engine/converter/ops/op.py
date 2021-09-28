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
from collections import namedtuple, OrderedDict
from .tensor import Tensor
from ..tf_utils import tf_extract_operator
from ..onnx_utils import onnx_extract_operator

OP_EXTRACTORS = {
    'tensorflow': tf_extract_operator,
    'onnxruntime': onnx_extract_operator,
}

OPERATORS = {}


def operator_registry(operator_type):
    """The class decorator used to register all Algorithm subclasses.

    Args:
        cls (class): The class of register.
        operator_type (str): The operator registration name

    Returns:
        cls: The class of register.
    """
    def decorator_operator(cls):
        if operator_type in OPERATORS:
            raise ValueError('Cannot have two operators with the same name')
        OPERATORS[operator_type] = cls
        return cls

    return decorator_operator


class Operator(object):
    def __init__(self):
        self._name = ''
        self._op_type = ''
        self._input_tensors = []
        self._output_tensors = []
        self._attr = OrderedDict()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def op_type(self):
        return self._op_type

    @op_type.setter
    def op_type(self, op_type):
        self._op_type = op_type

    @property
    def input_tensors(self):
        return self._input_tensors

    @input_tensors.setter
    def input_tensors(self, input_tensors):
        self._input_tensors = input_tensors

    @property
    def output_tensors(self):
        return self._output_tensors

    @output_tensors.setter
    def output_tensors(self, output_tensors):
        self._output_tensors = output_tensors

    @property
    def attr(self):
        return self._attr

    @attr.setter
    def attr(self, attr):
        self._attr = attr

    def set_attr(self, framework, node):
        self._attr = OrderedDict()

    # extract the op from framework
    def extract(self, framework, node, model, nodes_dict):
        self._name = node.name
        self._op_type, self._input_tensors, self._output_tensors = OP_EXTRACTORS[framework](
            node, model, nodes_dict)
        self.set_attr(framework, node)

    # make the op by set the attributes
    def construct(self, name, op_type, input_tensors=[], output_tensors=[], attr=OrderedDict()):
        self._name = name
        self._op_type = op_type
        self._input_tensors = input_tensors
        self._output_tensors = output_tensors
        self._attr = attr

    # get the op config in graph
    @property
    def config(self):
        conf_dict = OrderedDict()
        # conf_dict['type'] = self._op_type
        conf_dict['type'] = self._op_type
        if len(self._input_tensors) > 0:
            conf_dict['input'] = OrderedDict()
            for input_tensor in self._input_tensors:
                if self._op_type == 'Input':
                    conf_dict['input'][input_tensor.name] = input_tensor.config
                else:
                    conf_dict['input'][input_tensor.name] = {}
        if len(self._output_tensors) > 0:
            conf_dict['output'] = OrderedDict()
            for output_tensor in self._output_tensors:
                if self._op_type == 'Input':
                    conf_dict['output'][output_tensor.name] = output_tensor.config
                else:
                    conf_dict['output'][output_tensor.name] = {}

        if isinstance(self._attr, dict) and len(self._attr.keys()) > 0:
            conf_dict['attr'] = self._attr

        return conf_dict
