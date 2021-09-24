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

class Metric(object):
    """common Metric just collect the infos to construct a Metric
    """
    def __init__(self, metric_cls, name='user_metric', **kwargs):
        """The metric class should take the outputs of the model as the metric's inputs,
           neural_compressor built-in metric always take (predictions, labels) as inputs, it's
           recommended to design metric_cls to take (predictions, labels) as inputs.
           metric_cls should be sub_class of neural_compressor.metric.BaseMetric.
        """
        self.metric_cls = metric_cls
        self.name = name
        self.kwargs = kwargs

