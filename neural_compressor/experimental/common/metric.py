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
"""Common Metric just collects the information to construct a Metric."""
from deprecated import deprecated


@deprecated(version="2.0")
class Metric(object):
    """A wrapper of the information needed to construct a Metric.

    The metric class should take the outputs of the model as the metric's inputs,
    neural_compressor built-in metric always take (predictions, labels) as inputs, it's
    recommended to design metric_cls to take (predictions, labels) as inputs.
    """

    def __init__(self, metric_cls, name="user_metric", **kwargs):
        """Initialize a Metric with needed information.

        Args:
            metric_cls (cls): Should be a sub_class of neural_compressor.metric.BaseMetric,
                which takes (predictions, labels) as inputs
            name (str, optional): Name for metric. Defaults to 'user_metric'.
        """
        self.metric_cls = metric_cls
        self.name = name
        self.kwargs = kwargs
