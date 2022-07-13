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
import os
import shutil

from collections.abc import Iterable
from .nas_utils import find_pareto_front, NASMethods
from .search_algorithms import BayesianOptimizationSearcher, GridSearcher, RandomSearcher
from neural_compressor.conf.config import Conf, NASConfig
from neural_compressor.utils.utility import logger, LazyImport

torch = LazyImport('torch')


class NAS(object):
    def __new__(self, conf_fname_or_obj, *args, **kwargs):
        if isinstance(conf_fname_or_obj, str):
            if os.path.isfile(conf_fname_or_obj):
                self.conf = Conf(conf_fname_or_obj).usr_cfg
        elif isinstance(conf_fname_or_obj, NASConfig):
            self.conf = conf_fname_or_obj.usr_cfg
        else: # pragma: no cover
            raise NotImplementedError(
                "Please provide a str path to the config file."
            )
        assert self.conf.nas is not None, "nas section must be set"
        if isinstance(self.conf.nas.approach, str) and \
            self.conf.nas.approach.lower() in NASMethods:
            method = self.conf.nas.approach.lower()
        else:
            logger.warning(
                "NAS approach not set in config, use default NAS approach, i.e. Basic."
            )
            method = 'basic'
        return NASMethods[method](conf_fname_or_obj, *args, **kwargs)


class NASBase(object):
    """

    Args:
        search_space (dict): A dictionary for defining the search space.
        model_builder (function obj): A function to build model instance with the specified
            model architecture parameters.

    """

    def __init__(self, search_space=None, model_builder=None):
        super().__init__()
        self._search_space = search_space
        self._model_builder = model_builder
        self._search_algorithm = None
        self.search_results = {}
        self.best_model_archs = None
        self.seed = None

    def select_model_arch(self):
        """Propose architecture of the model based on search algorithm for next search iteration.

        Returns:
            Model architecture description.
        """
        model_arch_paras = self._search_algorithm.suggest()
        assert self.search_space_keys and isinstance(model_arch_paras, dict) and \
            self.search_space_keys == list(model_arch_paras.keys()), \
            "Keys of model_arch_paras should be the same with search_space_keys."
        return model_arch_paras

    def search(self, res_save_path=None):
        """NAS search process.

        Returns:
            Best model architecture found in search process.
        """
        assert self.model_builder is not None, \
            "Must specify model_builder for generating model instance by model architecture."
        if res_save_path is None or not os.path.isdir(res_save_path):
            res_save_path = os.getcwd()
        save_path = os.path.join(res_save_path, 'NASResults')
        self.model_paras_num = {}
        self.load_search_results(save_path)
        os.makedirs(save_path, exist_ok=True)

        for i in range(self.max_trials):
            logger.info(
                "{fix} Trial {n} starts, {r} trials to go {fix}".format(
                    n=i+1, r=self.max_trials-i-1, fix="="*30
                )
            )
            model_arch_paras = self.select_model_arch()
            logger.info("Model architecture {} proposed.".format(model_arch_paras))
            model = self._model_builder(model_arch_paras)
            model_paras = self.count_model_parameters(model)
            logger.info(
                "***** Number of model parameters: {:.2f}M *****".format(model_paras / 10**6)
            )
            self.model_paras_num[tuple(model_arch_paras.values())] = model_paras
            if tuple(model_arch_paras.values()) in self.search_results:
                logger.info("Skip evaluated model architecture {}.".format(model_arch_paras))
                continue
            if tuple(model_arch_paras.values()) in self.resumed_search_results:
                logger.info(
                    "Find previous results of model architecture: {}.".format(model_arch_paras)
                )
                metrics = self.resumed_search_results[tuple(model_arch_paras.values())]
            else:
                logger.info("Assessing model architecture: {}.".format(model_arch_paras))
                metrics = self.estimate(model)
            logger.info(
                "Metrics of model architecture {} is {}.".format(model_arch_paras, metrics)
            )
            self.search_results[tuple(model_arch_paras.values())] = metrics
            self._search_algorithm.get_feedback(sum(self.metrics_conversion(metrics)))
            self.dump_search_results(
                os.path.join(save_path, 'Trial_{}_results.txt'.format(i+1))
            )

        for model_arch_vec in self.resumed_search_results:
            if model_arch_vec not in self.search_results:
                self.search_results[model_arch_vec] = \
                    self.resumed_search_results[model_arch_vec]
                model = self._model_builder(self.params_vec2params_dict(model_arch_vec))
                self.model_paras_num[model_arch_vec] = self.count_model_parameters(model)
        self.dump_search_results(os.path.join(save_path, 'Final_results.txt'.format(i+1)))
        self.find_best_model_archs()
        logger.info(
            "{fix} Found {n} best model architectures {fix}".format(
                n=len(self.best_model_archs), fix="="*30
            )
        )
        for i, model_arch in enumerate(self.best_model_archs):
            logger.info("Best model architecture {}: {}".format(i+1, model_arch))
        return self.best_model_archs

    def estimate(self, model): # pragma: no cover
        """Estimate performance of the model. Depends on specific NAS algorithm.

        Returns:
            Evaluated metrics of the model.
        """
        raise NotImplementedError("Depends on specific NAS algorithm.")

    def count_model_parameters(self, model):
        if isinstance(model, torch.nn.Module):
            return sum(p.numel() for p in model.parameters())
        else:
            raise NotImplementedError("Only support torch model now.") # pragma: no cover

    def load_search_results(self, path):
        self.resumed_search_results = {}
        lastest_results_record = os.path.join(path, 'lastest_results.npy')
        if not os.path.exists(path) or not os.path.exists(lastest_results_record):
            return
        self.resumed_search_results = np.load(lastest_results_record, allow_pickle=True).item()
        os.makedirs(os.path.join(path, 'previous_results'), exist_ok=True)
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)):
                shutil.move(os.path.join(path, f), os.path.join(path, 'previous_results', f))
        logger.info("Loaded previous results.")

    def dump_search_results(self, path):
        lastest_results_record = os.path.join(os.path.dirname(path), 'lastest_results.npy')
        np.save(lastest_results_record, self.search_results, allow_pickle=True)
        write_contents = '=' * 30 + ' All Search Results ' + '=' * 30 + '\n\n'
        for model_arch_vec in self.search_results:
            tmp = ','.join(['{}_{}'.format(k, v) \
                for k, v in zip(self.search_space_keys, model_arch_vec)])
            write_contents += '{}: {} Paras: {}M\n'.format(
                tmp, self.search_results[model_arch_vec],
                self.model_paras_num[model_arch_vec] / 10**6
            )
        write_contents += '\n\n\n' + '=' * 30 + ' Best Search Results ' + '=' * 30 + '\n\n'
        self.find_best_model_archs()
        for i, model_arch in enumerate(self.best_model_archs):
            model_arch_vec = tuple(model_arch.values())
            tmp = ','.join(['{}_{}'.format(k, v) \
                for k, v in zip(self.search_space_keys, model_arch_vec)])
            write_contents += \
                '{}. {}: {} Paras: {}M\n'.format(
                    i+1, tmp, self.search_results[model_arch_vec],
                    self.model_paras_num[model_arch_vec] / 10**6
            )
        with open(path, mode='w') as f:
            f.write(write_contents)

    def params_vec2params_dict(self, paras_vec):
        assert len(paras_vec) == len(self.search_space_keys), \
            "Length of paras_vec and search_space_keys should be the same."
        return {k:v for k, v in zip(self.search_space_keys, paras_vec)}

    def find_best_model_archs(self):
        assert len(self.search_results) > 0, "Zero result in search_results."
        model_arches = list(self.search_results.keys())
        metrics = [self.metrics_conversion(self.search_results[ma]) for ma in model_arches]
        pareto_front_indices = find_pareto_front(metrics)
        self.best_model_archs = [self.params_vec2params_dict(model_arches[i]) \
            for i in pareto_front_indices]

    def metrics_conversion(self, metrics):
        if not isinstance(metrics, Iterable):
            metrics = [metrics]
        if isinstance(metrics, dict):
            if self.metrics is None:
                self.metrics = list(metrics.keys())
            assert list(metrics.keys()) == list(self.metrics), \
                "Keys of metrics not match with metrics in the configuration."
            metrics = list(metrics.values())
        if self.higher_is_better is None:
            self.higher_is_better = [True,] * len(metrics)
            logger.warning("higher_is_better not set in the configuration, " + \
                "set it to all True for every metric entry by default.")
        converted_metrics = [metric if higher_is_better else -metric \
            for metric, higher_is_better in zip(metrics, self.higher_is_better)]
        return converted_metrics

    def init_search_cfg(self, config):
        self.search_cfg = config.search

        if not self._search_space:
            self._search_space = self.search_cfg.search_space
        else:
            logger.warning(
                "Use user provided search space {}, instead of search space "
                "defined in the config, i.e. {}.".format(
                    self._search_space, self.search_cfg.search_space
                )
            )
        assert isinstance(self._search_space, dict) and len(self._search_space) > 0, \
            "Must provide a dict as search_space for NAS."
        self.search_space_keys = sorted(self.search_space.keys())
        for k in self.search_space_keys:
            assert isinstance(self.search_space[k], (list, tuple)), \
                "Value of key \'{}\' must be a list or tuple".format(k)

        self.metrics = self.search_cfg.metrics \
            if self.search_cfg.metrics else None
        self.higher_is_better = self.search_cfg.higher_is_better \
            if self.search_cfg.higher_is_better else None
        self.seed = self.search_cfg.seed
        self.max_trials = self.search_cfg.max_trials \
            if self.search_cfg.max_trials is not None else 3 # set default 3 for max_trials
        self.search_algorithm_type = self.search_cfg.search_algorithm \
            if self.search_cfg.search_algorithm else None
        if not self.search_algorithm_type:
            self._search_algorithm = BayesianOptimizationSearcher(self.search_space, self.seed)
        elif self.search_algorithm_type.lower() == 'grid':
            self._search_algorithm = GridSearcher(self.search_space)
        elif self.search_algorithm_type.lower() == 'random':
            self._search_algorithm = RandomSearcher(self.search_space, self.seed)
        elif self.search_algorithm_type.lower() == 'bo':
            self._search_algorithm = BayesianOptimizationSearcher(self.search_space, self.seed)
        else: # pragma: no cover
            logger.warning(
                'Please be aware that \'{}\' is not a built-in search algorithm.'.format(
                    self.search_algorithm_type
                )
            )

    @property
    def search_space(self):
        return self._search_space

    @search_space.setter
    def search_space(self, search_space):
        self._search_space = search_space

    @property
    def search_algorithm(self):
        return self._search_algorithm

    @search_algorithm.setter
    def search_algorithm(self, search_algorithm):
        self._search_algorithm = search_algorithm

    @property
    def model_builder(self):
        return self._model_builder

    @model_builder.setter
    def model_builder(self, model_builder):
        self._model_builder = model_builder

    def __repr__(self):
        return 'Base Class of NAS' # pragma: no cover