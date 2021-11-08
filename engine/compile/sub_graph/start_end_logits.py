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
import copy
from .. import graph_utils as util


@pattern_registry(pattern_type='StartEndLogits')
class StartEndLogits(Pattern):
    def __call__(self, model):

        patterns = {
            'StartEndLogits': [
                [[(0, 'Transpose'), (1, 'Unpack'), (2, 'Identity')]],
            ]
        }

        # bert_large_sparse_model
        in_pattern = patterns['StartEndLogits'][0]
        match_result = util.search_pattern(in_pattern, model)
        if len(match_result) != 0:
            for each_ret in match_result:
                model.remove_nodes(each_ret[:-1])

        return model
