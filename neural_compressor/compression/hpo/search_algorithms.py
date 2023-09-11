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
import xgboost as xgb

from neural_compressor.strategy.bayesian import BayesianOptimization

from ...config import HPOConfig
from .sa_optimizer import SimulatedAnnealingOptimizer
from .search_space import BaseSearchSpace, ContinuousSearchSpace, DiscreteSearchSpace

try:
    from neural_compressor.utils import logger
except:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


SEARCHERS = {}


def prepare_hpo(config):
    assert isinstance(config, HPOConfig), f"config should be {HPOConfig.__name__}"
    assert config.searcher in SEARCHERS.keys(), f"current only support search algorithms: {SEARCHERS.keys()}"
    if config.searcher == "xgb":
        return SEARCHERS[config.searcher](
            config.search_space,
            higher_is_better=config.higher_is_better,
            loss_type=config.loss_type,
            min_train_samples=config.min_train_samples,
            seed=config.seed,
        )
    else:
        return SEARCHERS[config.searcher](config.search_space)


def register_searcher(name):
    """Class decorator to register a Searcher subclass to the registry.

    Decorator function used before a Pattern subclass.
    Make sure that the Searcher class decorated by this function can be registered in SEARCHERS.

    Args:
        cls (class): The subclass of register.
        name: A string. Define the searcher type.

    Returns:
        cls: The class of register.
    """

    def register(searcher):
        SEARCHERS[name] = searcher
        return searcher

    return register


class Searcher(object):
    """Base class for defining the common methods of different search algorithms.

    Args:
        search_space (dict): A dictionary for defining the search space.
    """

    def __init__(self, search_space):
        assert isinstance(search_space, dict) and search_space, "Expect search_space to be a dict."
        self.search_space = search_space
        self.search_space_keys = sorted(search_space.keys())
        self.search_space_pool = self._create_search_space_pool()
        self.best = None
        for k in self.search_space_keys:
            assert isinstance(
                self.search_space[k], (list, tuple, BaseSearchSpace)
            ), "Value of key '{}' must be a list, tuple,\
                      CountinuousSearchSpace or DiscreteSearchSpace to specify choices".format(
                k
            )

    def _create_search_space_pool(self):
        """Build the search space pool."""
        search_space_pool = []
        for key in self.search_space_keys:
            if isinstance(self.search_space[key], (list, tuple)):
                space = DiscreteSearchSpace(value=self.search_space[key])
            else:  # pragma: no cover
                space = self.search_space[key]
            search_space_pool.append(space)
        return search_space_pool

    def suggest(self):
        """Suggest the model hyperparameter."""
        raise NotImplementedError("Depends on specific search algorithm.")  # pragma: no cover

    def get_feedback(self, metric):
        """Get metric feedback for the search algorithm."""
        pass

    def params_vec2params_dict(self, para_vec):
        """Convert the parameters vector to parameters dictionary.

        Where parameters vector and parameters dictionary both define the model hyperparameter.

        Returns:
            Parameters dictionary defining the model hyperparameter.
        """
        assert len(para_vec) == len(
            self.search_space_keys
        ), "Length of para_vec and search_space_keys should be the same."
        return {k: para_vec[i] for i, k in enumerate(self.search_space_keys)}


@register_searcher("grid")
class GridSearcher(Searcher):
    """Grid search.

    Search the whole search space exhaustively.

    Args:
        search_space (dict): A dictionary for defining the search space.
    """

    def __init__(self, search_space):
        """Initialize the attributes."""
        super().__init__(search_space)

        for space in self.search_space_pool:
            if space.type == "continuous":
                raise TypeError("GridSearcher not support continuous datatype, please use other algorithm.")

        self.idx = [0] * len(self.search_space_pool)

    def _add_idx(self, idx=0):
        def _add():
            if self.idx[idx] + 1 >= self.search_space_pool[idx].total_num:
                return False
            else:
                self.idx[idx] += 1
                return True

        if idx + 1 == len(self.idx):
            return _add()
        if self._add_idx(idx + 1):
            return True
        else:
            return _add()

    def suggest(self):
        """Suggest the model hyperparameter.

        Returns:
            The model hyperparameter.
        """
        param = []
        for i in range(len(self.idx)):
            param.append(self.search_space_pool[i].get_value(self.idx[i]))
        if not self._add_idx():  # pragma: no cover
            logger.warning("run out of search space pool, rebuild...")
            self.idx = [0] * len(self.search_space_pool)
        return self.params_vec2params_dict(param)


@register_searcher("random")
class RandomSearcher(Searcher):
    """Random search.

    Search the whole search space randomly.

    Args:
        search_space (dict): A dictionary for defining the search space.
    """

    def __init__(self, search_space):
        """Initialize the attributes."""
        super().__init__(search_space)

    def suggest(self):
        """Suggest the model hyperparameter.

        Returns:
            The model hyperparameter.
        """
        param = [s.get_value() for s in self.search_space_pool]
        return self.params_vec2params_dict(param)


@register_searcher("bo")
class BayesianOptimizationSearcher(Searcher):
    """Bayesian Optimization.

    Search the search space with Bayesian Optimization.

    Args:
        search_space (dict): A dictionary for defining the search space.
    """

    def __init__(self, search_space, seed=42):
        """Initialize the attributes."""
        super().__init__(search_space)
        idx_search_space = {}
        for key, space in zip(self.search_space_keys, self.search_space_pool):
            if isinstance(space, ContinuousSearchSpace):
                idx_search_space[key] = tuple(space.bound)
            else:
                idx_search_space[key] = (0, space.total_num - 1)
        self.bo_agent = BayesianOptimization(idx_search_space, random_seed=seed)

    def suggest(self):
        """Suggest the model hyperparameter.

        Returns:
            The model hyperparameter.
        """
        param_indices = self.bo_agent.gen_next_params()
        self.last_param_indices = param_indices
        return self.params_vec2params_dict(self.indices2params_vec(param_indices))

    def get_feedback(self, metric):
        """Get metric feedback and register this metric."""
        assert self.last_param_indices is not None, (
            "Need run suggest first " + "to get parameters and the input metric is corresponding to this parameters."
        )
        try:
            self.bo_agent._space.register(self.last_param_indices, metric)
        except KeyError:  # pragma: no cover
            logger.debug("Find registered params, skip it.")
            pass
        if self.best is None or self.best[1] < metric:
            param = self.params_vec2params_dict(self.indices2params_vec(self.last_param_indices))
            self.best = (param, metric)
        self.last_param_indices = None

    def feedback(self, param, metric):
        if self.best is None or self.best[1] < metric:
            self.best = (param, metric)
        self.bo_agent._space.register(param, metric)

    def indices2params_vec(self, indices):
        """Convert indices to parameters vector."""
        res = []
        for key, ind in indices.items():
            # keep ind within the index range of self.search_space[key]
            space = self.search_space_pool[self.search_space_keys.index(key)]
            if isinstance(space, ContinuousSearchSpace):
                res.append(ind)
            else:
                ind = int(min(max(round(ind), 0), space.total_num - 1))
                res.append(space.get_value(ind))
        return res


@register_searcher("xgb")
class XgbSearcher(Searcher):
    """XGBoost searcher.

    Search the search space with XGBoost model.

    Args:
        search_space (dict): A dictionary for defining the search space.
    """

    def __init__(self, search_space, higher_is_better=True, loss_type="reg", min_train_samples=10, seed=42):
        """Initialize the attributes."""
        super().__init__(search_space)

        self.seed = seed
        self.loss_type = loss_type
        self.higher_is_better = higher_is_better
        self.min_train_samples = min_train_samples
        self.log = {}

        self.last_params = None
        self._x = []
        self._y = []
        if loss_type == "reg":
            self.model = xgb.XGBRegressor(
                max_depth=3,
                n_estimators=100,
                gamma=0.0001,
                min_child_weight=1,
                subsample=1.0,
                eta=0.3,
                reg_lambda=1.00,
                reg_alpha=0,
                objective="reg:squarederror",
            )
        elif loss_type == "rank":  # pragma: no cover
            self.model = xgb.XGBRanker(
                max_depth=3,
                n_estimators=100,
                gamma=0.0001,
                min_child_weight=1,
                subsample=1.0,
                eta=0.3,
                reg_lambda=1.00,
                reg_alpha=0,
                objective="rank:pairwise",
            )
        else:  # pragma: no cover
            raise RuntimeError("Invalid loss type: {}, only support reg and rank".format(loss_type))
        self.optimizer = SimulatedAnnealingOptimizer(
            generate_func=self._generate_new_points, T0=100, Tf=0, alpha=0.9, higher_is_better=self.higher_is_better
        )

    def _generate_new_points(self, points):
        new_points = []
        for _ in range(len(points)):
            new_points.append([s.get_value() for s in self.search_space_pool])
        return new_points

    def suggest(self):
        """Suggest the model hyperparameter.

        Returns:
            The model hyperparameter.
        """
        if len(self._y) < self.min_train_samples:
            params = [s.get_value() for s in self.search_space_pool]
        else:
            x_train, y_train = np.array(self._x), np.array(self._y)

            self.model.fit(x_train, y_train)
            params = self.optimizer.gen_next_params(self.model.predict, self._x)

        self.last_params = params
        return self.params_vec2params_dict(params)

    def get_feedback(self, metric):
        """Get metric feedback and register this metric."""
        assert self.last_params is not None, (
            "Need run suggest first " + "to get parameters and the input metric is corresponding to this parameters."
        )
        if self.best is None or self.best[1] < metric:
            self.best = (self.params_vec2params_dict(self.last_params), metric)
        self._x.append(self.last_params)
        self._y.append(metric)
        params_key = "_".join([str(x) for x in self.last_params])
        self.log[params_key] = metric
        self.last_params = None

    def feedback(self, param, metric):
        param_list = []
        for k in self.search_space_keys:
            param_list.append(param[k])
        if self.best is None or self.best[1] < metric:
            self.best = (param, metric)
        self._x.append(param_list)
        self._y.append(metric)
        params_key = "_".join([str(x) for x in param])
        self.log[params_key] = metric
