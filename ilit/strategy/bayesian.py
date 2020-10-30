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

from .strategy import strategy_registry, TuneStrategy
import warnings
import numpy as np
import copy
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from ..utils import logger

@strategy_registry
class BayesianTuneStrategy(TuneStrategy):
    """The tuning strategy using bayesian search in tuning space.

    Args:
        model (object):                        The FP32 model specified for low precision tuning.
        conf (Conf):                           The Conf class instance initialized from user yaml
                                               config file.
        q_dataloader (generator):              Data loader for calibration, mandatory for
                                               post-training quantization.
                                               It is iterable and should yield a tuple (input,
                                               label) for calibration dataset containing label,
                                               or yield (input, _) for label-free calibration
                                               dataset. The input could be a object, list, tuple or
                                               dict, depending on user implementation, as well as
                                               it can be taken as model input.
        q_func (function, optional):           Reserved for future use.
        eval_dataloader (generator, optional): Data loader for evaluation. It is iterable
                                               and should yield a tuple of (input, label).
                                               The input could be a object, list, tuple or dict,
                                               depending on user implementation, as well as it can
                                               be taken as model input. The label should be able
                                               to take as input of supported metrics. If this
                                               parameter is not None, user needs to specify
                                               pre-defined evaluation metrics through configuration
                                               file and should set "eval_func" paramter as None.
                                               Tuner will combine model, eval_dataloader and
                                               pre-defined metrics to run evaluation process.
        eval_func (function, optional):        The evaluation function provided by user.
                                               This function takes model as parameter, and
                                               evaluation dataset and metrics should be
                                               encapsulated in this function implementation and
                                               outputs a higher-is-better accuracy scalar value.

                                               The pseudo code should be something like:

                                               def eval_func(model):
                                                    input, label = dataloader()
                                                    output = model(input)
                                                    accuracy = metric(output, label)
                                                    return accuracy
        dicts (dict, optional):                The dict containing resume information.
                                               Defaults to None.

    """

    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None):
        self.bayes_opt = None
        super(
            BayesianTuneStrategy,
            self).__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts)

    def __getstate__(self):
        for history in self.tuning_history:
            if self._same_yaml(history['cfg'], self.cfg):
                history['bayes_opt'] = self.bayes_opt
        save_dict = super(BayesianTuneStrategy, self).__getstate__()
        return save_dict

    def params_to_tune_configs(self, params):
        op_cfgs = {}
        op_cfgs['op'] = {}
        for op, configs in self.opwise_quant_cfgs.items():
            if len(configs) > 1:
                value = int(params[op[0]])
                if value == len(configs):
                    value = len(configs) - 1
                op_cfgs['op'][op] = copy.deepcopy(configs[value])
            elif len(configs) == 1:
                op_cfgs['op'][op] = copy.deepcopy(configs[0])
            else:
                op_cfgs['op'][op] = copy.deepcopy(self.opwise_tune_cfgs[op][0])
        if len(self.calib_iter) > 1:
            value = int(params['calib_iteration'])
            if value == len(self.calib_iter):
                value = len(configs) - 1
            op_cfgs['calib_iteration'] = int(self.calib_iter[value])
        else:
            op_cfgs['calib_iteration'] = int(self.calib_iter[0])
        return op_cfgs

    def next_tune_cfg(self):
        """The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.

        """
        params = None
        pbounds = {}
        for op, configs in self.opwise_quant_cfgs.items():
            if len(configs) > 1:
                pbounds[op[0]] = (0, len(configs))
        if len(self.calib_iter) > 1:
            pbounds['calib_iteration'] = (0, len(self.calib_iter))
        if len(pbounds) == 0:
            yield self.params_to_tune_configs(params)
            return
        if self.bayes_opt is None:
            self.bayes_opt = BayesianOptimization(
                pbounds=pbounds, random_seed=self.cfg.tuning.random_seed)
        while True:
            params = self.bayes_opt.gen_next_params()
            logger.debug("Current params are: %s" % params)
            yield self.params_to_tune_configs(params)
            try:
                self.bayes_opt._space.register(params, self.last_tune_result[0])
            except KeyError:
                logger.debug("This params has been registered before, will skip it!")
                pass

# Util part
# Bayesian opt acq function


def acq_max(ac, gp, y_max, bounds, random_seed, n_warmup=10000, n_iter=10):
    """
    A function to find the maximum of the acquisition function
    Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    random_state: instance of np.RandomState random number generator
    n_warmup: number of times to randomly sample the aquisition function
    n_iter: number of times to run scipy.minimize
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))

# Target space part


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added
    """

    def __init__(self, pbounds, random_seed=9527):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.
        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.
        random_seed : int
            optionally specify a seed for a random number generator
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
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
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register(self, params, target):
        """
        Append a point and its target value to the known data.
        Parameters
        ----------
        params: ndarray
            a single point, with len(params) == self.dim
        target: float
            target function value
        Raises
        ------
        KeyError:
            if the point is not unique
        Notes
        -----
        runs in ammortized constant time
        """
        x = self._as_array(params)
        if x in self:
            raise KeyError('Params point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def get_target(self, params):
        """
        Get the target value of params
        ----------
        params: ndarray
            a single point, with len(params) == self.dim
        Returns
        -------
        target: float
            target function value.
        """
        x = self._as_array(params)
        target = self._cache[_hashable(x)]
        return target

    def random_sample(self):
        """
        Creates random points within the bounds of the space.
        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = np.random.uniform(  # pylint: disable=unsupported-assignment-operation
                lower, upper, size=1)  
        return data.ravel()

    def max(self):
        """Get maximum target value found and corresponding parametes."""
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def res(self):
        """Get all target values found and corresponding parametes."""
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]

# Tuning part


class BayesianOptimization():
    def __init__(self, pbounds, random_seed=9527, verbose=2):
        self._random_seed = random_seed
        np.random.seed(self._random_seed)
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
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    @staticmethod
    def _ucb(x, gp, y_max, kappa=2.576):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        return mean + kappa * std

    def suggest(self):
        """Most promissing point to probe next"""
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
            random_seed=self._random_seed
        )
        return self._space.array_to_params(suggestion)

    def gen_next_params(self):
        next_params = self.suggest()
        return next_params
