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
"""The Bayesian tuning strategy."""

import warnings
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from ..config import options
from ..utils import logger
from .strategy import TuneStrategy, strategy_registry
from .utils.tuning_sampler import OpWiseTuningSampler


@strategy_registry
class BayesianTuneStrategy(TuneStrategy):
    """The Bayesian tuning strategy."""

    def __init__(
        self,
        model,
        conf,
        q_dataloader=None,
        q_func=None,
        eval_func=None,
        eval_dataloader=None,
        eval_metric=None,
        resume=None,
        q_hooks=None,
    ):
        """Init the BaySian tuning strategy.

        Args:
            model: The FP32 model specified for low precision tuning.
            conf: The Conf class instance includes all user configurations.
            q_dataloader: Data loader for calibration, mandatory for post-training quantization.  Defaults to None.
            q_func: Training function for quantization aware training. Defaults to None. Defaults to None.
            eval_func: The evaluation function provided by user. This function takes model as parameter, and
                evaluation dataset and metrics should be encapsulated in this function implementation and
                outputs a higher-is-better accuracy scalar value.
            eval_dataloader: Data loader for evaluation. Defaults to None.
            eval_metric: Metric for evaluation. Defaults to None.
            resume: The dict containing resume information. Defaults to None.
            q_hooks: The dict of training hooks, supported keys are: on_epoch_begin, on_epoch_end, on_step_begin,
                on_step_end. Their values are functions to be executed in adaptor layer.. Defaults to None.
        """
        super().__init__(
            model=model,
            conf=conf,
            q_dataloader=q_dataloader,
            q_func=q_func,
            eval_func=eval_func,
            eval_dataloader=eval_dataloader,
            eval_metric=eval_metric,
            resume=resume,
            q_hooks=q_hooks,
        )
        self.bayes_opt = None

    def __getstate__(self):
        """Magic method for pickle saving.

        Returns:
            dict: Saved dict for resuming
        """
        for history in self.tuning_history:
            if self._same_conf(history["cfg"], self.conf):
                history["bayes_opt"] = self.bayes_opt
        save_dict = super().__getstate__()
        return save_dict

    def _params_to_tune_configs(self, params):
        op_tuning_cfg = {}
        calib_sampling_size_lst = self.tuning_space.root_item.get_option_by_name("calib_sampling_size").options
        for op_name_type, configs in self.op_configs.items():
            if len(configs) == 1:
                op_tuning_cfg[op_name_type] = configs[0]
            else:
                op_tuning_cfg[op_name_type] = configs[min(len(configs) - 1, int(params[op_name_type[0]]))]
        if len(calib_sampling_size_lst) > 1:
            calib_sampling_size = calib_sampling_size_lst[min(len(configs) - 1, int(params["calib_sampling_size"]))]
        else:
            calib_sampling_size = calib_sampling_size_lst[0]
        op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
        return op_tuning_cfg

    def next_tune_cfg(self):
        """Generate the next tuning config according to bayesian search algorithm.

        This strategy comes from the Bayesian optimization package and changed it to a discrete version.
        It uses Gaussian processes to define the prior/posterior distribution over the black-box
        function with the tuning history and then finds the tuning configuration that maximizes
        the expected improvement.

        Returns:
            tune_config (dict): A dict containing the tuning configuration for quantization.
        """
        params = None
        pbounds = {}
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name("calib_sampling_size").options
        op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg = self.initial_tuning_cfg()
        op_wise_pool = OpWiseTuningSampler(tuning_space, [], [], op_item_dtype_dict, initial_op_tuning_cfg)
        self.op_configs = op_wise_pool.get_opwise_candidate()

        for op_name_type, configs in self.op_configs.items():
            if len(configs) > 1:
                pbounds[op_name_type[0]] = (0, len(configs))
        if len(calib_sampling_size_lst) > 1:
            pbounds["calib_sampling_size"] = (0, len(calib_sampling_size_lst))
        if len(pbounds) == 0:
            yield self._params_to_tune_configs(params)
            return
        if self.bayes_opt is None:
            self.bayes_opt = BayesianOptimization(pbounds=pbounds, random_seed=options.random_seed)
        while True:
            params = self.bayes_opt.gen_next_params()
            logger.debug("Dump current bayesian params:")
            logger.debug(params)
            yield self._params_to_tune_configs(params)
            try:
                self.bayes_opt._space.register(params, self.last_tune_result[0])
            except KeyError:
                logger.debug("Find registered params, skip it.")


# Util part
# Bayesian opt acq function


def acq_max(ac, gp, y_max, bounds, random_seed, n_warmup=10000, n_iter=10):
    """Find the maximum of the acquisition function parameters.

    Args:
        ac: The acquisition function object that return its point-wise value.
        gp: A gaussian process fitted to the relevant data.
        y_max: The current maximum known value of the target function.
        bounds: The variables bounds to limit the search of the acq max.
        random_seed: instance of np.RandomState random number generator
        n_warmup: number of times to randomly sample the acquisition function
        n_iter: number of times to run scipy.minimize

    Returns:
        x_max: The arg max of the acquisition function.
    """
    # Warm up with random points
    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more thoroughly
    x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(
            lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max), x_try.flatten(), bounds=bounds, method="L-BFGS-B"
        )

        # See if success
        if not res.success:
            continue

        if isinstance(res.fun, float):
            res.fun = np.array([res.fun])
        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


def _hashable(x):
    """Ensure that an point is hashable by a python dict."""
    return tuple(map(float, x))


# Target space part
class TargetSpace(object):
    """Holds the param-space coordinates (X) and target values (Y).

    Allows for constant-time appends while ensuring no duplicates are added.
    """

    def __init__(self, pbounds, random_seed=9527):
        """Construct a TargetSpace.

        Args:
            target_func (function): Function to be maximized.
            pbounds (dict): Dictionary with parameters names as keys and a tuple with minimum and maximum values.
            random_seed (int): Optionally specify a seed for a random number generator
        """
        self.random_seed = random_seed
        # Get the name of the parameters
        names = list(pbounds.keys())
        self._keys = deepcopy(names)
        # Create an array with parameters bounds
        self._bounds = np.array([pbounds[name] for name in names], dtype=np.float32)

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x):
        """Check if param x is cached in this space."""
        return _hashable(x) in self._cache

    def __len__(self):
        """Get the total count of stored items."""
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        """Check if the space is empty."""
        return len(self) == 0

    @property
    def params(self):
        """Get all params stored in this space."""
        return self._params

    @property
    def target(self):
        """Get all target values in this space."""
        return self._target

    @property
    def dim(self):
        """Get the dimension of this space."""
        return len(self._keys)

    @property
    def keys(self):
        """Get all keys of this space."""
        return self._keys

    @property
    def bounds(self):
        """Get the bounds of this space."""
        return self._bounds

    def params_to_array(self, params):
        """Generate an array from params.

        Args:
            params (Dict): The dict contains keys in `self.keys`, and
              corresponding param.

        Returns:
            np.array: An array contains all params.
        """
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(list(params.keys()))
                + "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        """Generate an params' dict from array.

        Args:
            x (np.array): The array contains all params.

        Returns:
            dict: the dict contains keys and the params corresponding to it.
        """
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x))
                + "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x))
                + "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register(self, params, target):
        """Append a point and its target value to the known data.

        Runs in amortized constant time.

        Args:
            params (ndarray): a single point, with len(params) == self.dim
            target (float): target function value

        Raises:
            KeyError: if the point is not unique
        """
        x = self._as_array(params)
        if x in self:
            raise KeyError("Params point {} is not unique".format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def get_target(self, params):
        """Get the target value of params.

        Args:
            params (ndarray): a single point, with len(params) == self.dim

        Returns:
            target (float): target function value.
        """
        x = self._as_array(params)
        target = self._cache[_hashable(x)]
        return target

    def random_sample(self):
        """Create random points within the bounds of the space.

        Returns:
            data (ndarray): [num x dim] array points with dimensions corresponding to `self._keys`
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = np.random.uniform(lower, upper, size=1)  # pylint: disable=unsupported-assignment-operation
        return data.ravel()

    def max(self):
        """Get maximum target value found and corresponding parameters."""
        try:
            res = {"target": self.target.max(), "params": dict(zip(self.keys, self.params[self.target.argmax()]))}
        except ValueError:
            res = {}
        return res

    def res(self):
        """Get all target values found and corresponding parameters."""
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [{"target": target, "params": param} for target, param in zip(self.target, params)]


# Tuning part
class BayesianOptimization:
    """The class for bayesian optimization.

    This class takes the parameters bounds in order to find which values for
    the parameters yield the maximum value using bayesian optimization.
    """

    def __init__(self, pbounds, random_seed=9527, verbose=2):
        """Init bayesian optimization.

        Args:
            pbounds (dict): Dictionary with parameters names as keys and a tuple with
              minimum and maximum values.
            random_seed (int, optional): The seed for random searching. Default to 9527.
            verbose (int, optional): The level of verbosity. Default to 2.
        """
        self._random_seed = random_seed
        # Data structure containing the bounds of its domain,
        # and a record of the points we have evaluated.
        self._space = TargetSpace(pbounds, random_seed)

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_seed,
        )
        self._verbose = verbose

    @property
    def space(self):
        """Get the target space."""
        return self._space

    @property
    def max(self):
        """Get the maximum value of target space."""
        return self._space.max()

    @property
    def res(self):
        """Get the minimum value of target space."""
        return self._space.res()

    @staticmethod
    def _ucb(x, gp, y_max, kappa=2.576):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        return mean + kappa * std

    def suggest(self):
        """Suggest the most promising points."""
        if len(set(self._space.target)) < 2:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=self._ucb,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_seed=self._random_seed,
        )
        return self._space.array_to_params(suggestion)

    def gen_next_params(self):
        """Get the next parameter."""
        next_params = self.suggest()
        return next_params
