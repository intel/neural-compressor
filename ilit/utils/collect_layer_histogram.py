#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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

import numpy as np


class LayerHistogramCollector(object):
    """Saves layer histogram in a dict with layer names as keys and lists of NDArrays as
    values. The collected histogram will be used for calculating the optimal thresholds for
    quantization using KL divergence.
    """

    def __init__(self, num_bins=8001, layer_tensor=None, include_layer=None, logger=None):
        self.hist_dict = {}
        self.num_bins = num_bins
        self.layer_tensor = layer_tensor
        self.include_layer = include_layer
        self.logger = logger

    def collect(self):
        """Callback function for collecting layer output NDArrays."""
        # name = py_str(name)
        # if name not in self.include_layer:
        #     return
        # handle = ctypes.cast(arr, NDArrayHandle)
        # arr = NDArray(handle, writable=False).copyto(cpu()).asnumpy()
        for name in self.layer_tensor:
            if name not in self.include_layer:
                continue
            for arr in self.layer_tensor[name]:
                if self.logger:
                    self.logger.debug(
                        "Collecting layer %s histogram of shape %s" %
                        (name, arr.shape))
                min_range = np.min(arr)
                max_range = np.max(arr)
                th = max(abs(min_range), abs(max_range))
                if name in self.hist_dict:
                    self.hist_dict[name] = self.combine_histogram(
                        self.hist_dict[name], arr, min_range, max_range, th)
                else:
                    hist, hist_edges = np.histogram(
                        arr, bins=self.num_bins, range=(-th, th))
                    self.hist_dict[name] = (
                        hist, hist_edges, min_range, max_range, th)

    def combine_histogram(self, old_hist, arr, new_min, new_max, new_th):
        """ Collect layer histogram for arr and combine it with old histogram.
        """
        (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist
        if new_th <= old_th:
            hist, _ = np.histogram(arr, bins=len(old_hist), range=(-old_th, old_th))
            return (old_hist + hist, old_hist_edges,
                    min(old_min, new_min), max(old_max, new_max), old_th)
        else:
            # Need to generate new histogram with new_th
            old_num_bins = len(old_hist)
            old_step = 2 * old_th / old_num_bins
            half_increased_bins = int((new_th - old_th) // old_step + 1)
            new_num_bins = half_increased_bins * 2 + old_num_bins
            new_th = half_increased_bins * old_step + old_th
            hist, hist_edges = np.histogram(
                arr, bins=new_num_bins, range=(-new_th, new_th))
            hist[half_increased_bins:new_num_bins -
                 half_increased_bins] += old_hist
            return (
                hist, hist_edges, min(
                    old_min, new_min), max(
                    old_max, new_max), new_th)
