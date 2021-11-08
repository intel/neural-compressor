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

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='OutputData')
class OutputData(Pattern):
    def __call__(self, model):

        # make the output_data node in graph
        last_node = model.nodes[-1]
        input_tensors = copy.deepcopy(last_node.output_tensors)
        output_data_node = util.construct_node('output_data',
                                               'Output',
                                               input_tensors=input_tensors)
        model.insert_nodes(len(model.nodes), [output_data_node])
        model.nodes[-1].attr = None
        
        return model
