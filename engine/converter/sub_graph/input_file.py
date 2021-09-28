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


# remove the input_file patterns in tensorflow model if it has
@pattern_registry(pattern_type='InputFile')
class InputFile(Pattern):
    def __call__(self, model):

        patterns = {
            'InputFile': [
                [[(0, 'Placeholder'), (2, 'Reshape'), (3, 'TensorSliceDataset'),
                  (4, 'FlatMapDataset'), (5, 'MapAndBatchDataset'), (6, 'OptimizeDataset'),
                  (7, 'ModelDataset'), (9, 'MakeIterator')],
                 [(), (1, 'Placeholder'), (5, 'MapAndBatchDataset')],
                 [(), (8, 'IteratorV2'), (9, 'MakeIterator')]],
            ]
        }

        in_pattern = patterns['InputFile'][0]
        match_result = util.search_pattern(in_pattern, model)
        for each_ret in match_result:
            model.remove_nodes(each_ret[:-1])

        return model
