"""Search algorithms for NAS."""

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

import random
from .nas_utils import create_search_space_pool
from neural_compressor.strategy.bayesian import BayesianOptimization
from neural_compressor.utils import logger


class Searcher(object):
    """Base class for defining the common methods of different search algorithms.

    Args:
        search_space (dict): A dictionary for defining the search space.
    """

    def __init__(self, search_space) -> None:
        """Initialize the attributes."""
        assert isinstance(search_space, dict) and search_space, \
            "Expect search_space to be a dict."
        self.search_space = search_space
        self.search_space_keys = sorted(search_space.keys())
        for k in self.search_space_keys:
            assert isinstance(self.search_space[k], (list, tuple)), \
                "Value of key \'{}\' must be a list or tuple to specify choices".format(
                    k)

    def suggest(self):
        """Suggest the model architecture."""
        raise NotImplementedError('Depends on specific search algorithm.') # pragma: no cover

    def get_feedback(self, metric):
        """Get metric feedback for the search algorithm."""
        pass

    def params_vec2params_dict(self, para_vec):
        """Convert the parameters vector to parameters dictionary.

        Where parameters vector and parameters dictionary both define the model architecture.

        Returns:
            Parameters dictionary defining the model architecture.
        """
        assert len(para_vec) == len(self.search_space_keys), \
            "Length of para_vec and search_space_keys should be the same."
        return {k: para_vec[i] for i, k in enumerate(self.search_space_keys)}


class GridSearcher(Searcher):
    """Grid search.

    Search the whole search space exhaustively.

    Args:
        search_space (dict): A dictionary for defining the search space.
    """

    def __init__(self, search_space) -> None:
        """Initialize the attributes."""
        super(GridSearcher, self).__init__(search_space)
        self.search_space_pool = create_search_space_pool(search_space)
        self.idx = 0

    def suggest(self):
        """Suggest the model architecture.

        Returns:
            The model architecture.
        """
        res = self.search_space_pool[self.idx]
        self.idx = (self.idx + 1) % len(self.search_space_pool)
        return self.params_vec2params_dict(res)


class RandomSearcher(Searcher):
    """Random search.

    Search the whole search space randomly.

    Args:
        search_space (dict): A dictionary for defining the search space.
    """

    def __init__(self, search_space, seed=42) -> None:
        """Initialize the attributes."""
        super(RandomSearcher, self).__init__(search_space)
        self.search_space_pool = create_search_space_pool(search_space)
        self.indices_pool = list(range(len(self.search_space_pool)))
        random.seed(seed)
        random.shuffle(self.indices_pool)

    def suggest(self):
        """Suggest the model architecture.

        Returns:
            The model architecture.
        """
        if not self.indices_pool:
            self.indices_pool = list(range(len(self.search_space_pool)))
            random.shuffle(self.indices_pool)
        idx = self.indices_pool.pop(-1)
        return self.params_vec2params_dict(self.search_space_pool[idx])


class BayesianOptimizationSearcher(Searcher):
    """Bayesian Optimization.

    Search the search space with Bayesian Optimization.

    Args:
        search_space (dict): A dictionary for defining the search space.
    """

    def __init__(self, search_space, seed=42) -> None:
        """Initialize the attributes."""
        super(BayesianOptimizationSearcher, self).__init__(search_space)
        idx_search_space = {
            k: (0, len(search_space[k])-1) for k in self.search_space_keys}
        self.bo_agent = BayesianOptimization(
            idx_search_space, random_seed=seed)
        self.last_param_indices = None

    def suggest(self):
        """Suggest the model architecture.

        Returns:
            The model architecture.
        """
        param_indices = self.bo_agent.gen_next_params()
        self.last_param_indices = param_indices
        return self.params_vec2params_dict(self.indices2params_vec(param_indices))

    def get_feedback(self, metric):
        """Get metric feedback and register this metric."""
        assert self.last_param_indices is not None, "Need run suggest first " + \
            "to get parameters and the input metric is corresponding to this parameters."
        try:
            self.bo_agent._space.register(self.last_param_indices, metric)
        except KeyError:  # pragma: no cover
            logger.debug("Find registered params, skip it.")
            pass
        self.last_param_indices = None

    def indices2params_vec(self, indices):
        """Convert indices to parameters vector."""
        res = []
        for key, ind in indices.items():
            # keep ind within the index range of self.search_space[key]
            ind = int(min(max(round(ind), 0), len(self.search_space[key])-1))
            res.append(self.search_space[key][ind])
        return res
