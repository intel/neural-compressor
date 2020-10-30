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

import os
from .adaptor import FRAMEWORKS
from .objective import OBJECTIVES
from .conf.config import Conf
from .utils import logger
from .utils.create_obj_from_config import create_eval_func, create_dataset, create_dataloader
from .conf.dotdict import deep_get
from .data import DataLoader as DATALOADER

class Benchmark(object):
    """Benchmark class can be used to evaluate the model performance, with the objective
       setting, user can get the data of what they configured in yaml

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal,
        tuning objective and preferred calibration & quantization tuning space etc.

    """

    def __init__(self, conf_fname):
        self.conf = Conf(conf_fname)

    def __call__(self, model, b_dataloader=None, b_func=None):
        cfg = self.conf.usr_cfg
        framework_specific_info = {'device': cfg.device, \
                                   'approach': cfg.quantization.approach, \
                                   'random_seed': cfg.tuning.random_seed}
        framework = cfg.model.framework.lower()
        if framework == 'tensorflow':
            framework_specific_info.update({"inputs": cfg.model.inputs, \
                                            "outputs": cfg.model.outputs})
        if framework == 'mxnet':
            framework_specific_info.update({"b_dataloader": b_dataloader})

        adaptor = FRAMEWORKS[framework](framework_specific_info)

        assert cfg.evaluation is not None, 'benchmark need evaluation filed not be None'
        results = {}
        for mode in cfg.evaluation.keys():
            iteration = -1 if deep_get(cfg, 'evaluation.{}.iteration'.format(mode)) is None \
                else deep_get(cfg, 'evaluation.{}.iteration'.format(mode))
            metric =  deep_get(cfg, 'evaluation.{}.metric'.format(mode))

            if b_dataloader is None:
                assert deep_get(cfg, 'evaluation.{}.dataloader'.format(mode)) is not None, \
                    'dataloader field of yaml file is missing'

                b_dataloader_cfg = deep_get(cfg, 'evaluation.{}.dataloader'.format(mode))
                b_dataloader = create_dataloader(framework, b_dataloader_cfg)
                b_postprocess_cfg = deep_get(cfg, 'evaluation.{}.postprocess'.format(mode))
                b_func = create_eval_func(framework, \
                                          b_dataloader, \
                                          adaptor, \
                                          metric, \
                                          b_postprocess_cfg,
                                          iteration=iteration)
            else:
                b_func = create_eval_func(framework, \
                                          b_dataloader, \
                                          adaptor, \
                                          metric, \
                                          iteration=iteration)

            objective = cfg.tuning.objective.lower()
            self.objective = OBJECTIVES[objective](cfg.tuning.accuracy_criterion, \
                                                   is_measure=True)

            val = self.objective.evaluate(b_func, model)
            logger.info('{} mode benchmark done!'.format(mode))
            # measurer contain info not only performance(eg, memory, model_size)
            # also measurer have result list among steps
            acc, _ = val
            batch_size = b_dataloader.batch_size
            warmup =  0 if deep_get(cfg, 'evaluation.{}.warmup'.format(mode)) is None \
                else deep_get(cfg, 'evaluation.{}.warmup'.format(mode))

            assert len(self.objective.measurer.result_list()) > warmup, \
                'itreation should larger than warmup'

            results[mode] = acc, batch_size, \
                            self.objective.measurer.result_list()[warmup:]

        return results
