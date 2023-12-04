"""Common methods for NAS."""

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

import numpy as np
from deprecated import deprecated

NASMethods = {}


@deprecated(version="2.0")
def nas_registry(nas_method):
    """Decorate the NAS subclasses.

    The class decorator used to register all NAS subclasses.

    Args:
        nas_method (str): The string of supported NAS Method.

    Returns:
        cls: The class of register.
    """
    assert isinstance(nas_method, str), "Expect nas_method to be a string."

    def decorator(cls):
        NASMethods[nas_method.lower()] = cls
        return cls

    return decorator


@deprecated(version="2.0")
def create_search_space_pool(search_space, idx=0):
    """Create all the samples from the search space.

    Args:
        search_space (dict): A dict defining the search space.
        idx (int): An index for indicating which key of search_space to enumerate.

    Return:
        A list of all the samples from the search space.
    """
    search_space_keys = sorted(search_space.keys())
    if idx == len(search_space_keys):
        return [[]]
    key = search_space_keys[idx]
    search_space_pool = []
    for v in search_space[key]:
        sub_search_space_pool = create_search_space_pool(search_space, idx + 1)
        search_space_pool += [[v] + item for item in sub_search_space_pool]
    return search_space_pool


@deprecated(version="2.0")
def find_pareto_front(metrics):
    """Find the pareto front points, assuming all metrics are "higher is better".

    Args:
        metrics (numpy array or list): An (n_points, n_metrics) array.

    Return:
        An array of indices of pareto front points.
        It is a (n_pareto_points, ) integer array of indices.
    """
    metrics = np.array(metrics)
    pareto_front_point_indices = np.arange(metrics.shape[0])
    next_point_idx = 0
    while next_point_idx < len(metrics):
        nondominated_points = np.any(metrics > metrics[next_point_idx], axis=1)
        nondominated_points[next_point_idx] = True
        # Remove points being dominated by current point
        pareto_front_point_indices = pareto_front_point_indices[nondominated_points]
        metrics = metrics[nondominated_points]
        next_point_idx = np.sum(nondominated_points[: next_point_idx + 1])
    return pareto_front_point_indices
