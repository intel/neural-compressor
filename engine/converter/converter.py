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

"""
1. First use model loader to get the computation graph with corresponding framework.
   The graph contains nodes and edges, the node is op and the edge is the tensor.
2. Then extract the ops in the graph and pack them to our form.
3. Next exploit these above ops to consist sub-graph, which can see as "a new big op", like
   LayerNorm. Note that there may have different computation flow in one subgraph.
4. Finally, convert them to .yaml file and .bin file for model configuration and inference.
"""

import datetime
import logging as log
import os
import sys
import traceback
from collections import OrderedDict
from .loaders.loader import Loader
from .extractors.extractor import Extractor
from .sub_graph.subgraph_matcher import SubGraphMatcher


CONVERTERS = OrderedDict({
    'loader': Loader,
    'extractor': Extractor,
    'sub_graph': SubGraphMatcher,
})


def start_pipeline(model, config=None):
    converter_list = []
    # initialize the converter
    for converter_type in CONVERTERS.keys():
        converter = CONVERTERS[converter_type]()
        converter_list.append(converter)
    # convert the model
    for converter in converter_list:
        model = converter(model)
    return model


def prepare_ir(model, config=None):
    model = start_pipeline(model, config=None)
    return model
