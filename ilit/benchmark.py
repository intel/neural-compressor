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

from .adaptor import FRAMEWORKS
from .objective import OBJECTIVES
from .conf.config import Conf
from .utils import logger
from .utils.create_obj_from_config import create_eval_func, create_dataloader
from .conf.dotdict import deep_get, deep_set
from .data import DataLoader as DATALOADER
from .data import TRANSFORMS
from .metric import METRICS

class Benchmark(object):
    """Benchmark class can be used to evaluate the model performance, with the objective
       setting, user can get the data of what they configured in yaml

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal,
        tuning objective and preferred calibration & quantization tuning space etc.

    """

    def __init__(self, conf_fname):
        self.conf = Conf(conf_fname)
        self.framework = self.conf.usr_cfg.model.framework.lower()

    def __call__(self, model, b_dataloader=None, b_func=None):
        cfg = self.conf.usr_cfg
        framework_specific_info = {'device': cfg.device, \
                                   'approach': cfg.quantization.approach, \
                                   'random_seed': cfg.tuning.random_seed}
        framework = cfg.model.framework.lower()
        if framework == 'tensorflow':
            framework_specific_info.update({"inputs": cfg.model.inputs, \
                                            "outputs": cfg.model.outputs, \
                                            'workspace_path': cfg.tuning.workspace.path})
        if framework == 'mxnet':
            framework_specific_info.update({"b_dataloader": b_dataloader})
        if 'onnxrt' in framework.lower():
            framework_specific_info.update({"backend": framework.lower().split('_')[-1], \
                                            'workspace_path': cfg.tuning.workspace.path})
        if framework == 'pytorch':
            framework_specific_info.update({"q_dataloader": None,
                                            "benchmark": True})
        if framework == 'pytorch_ipex':
            framework_specific_info.update({"workspace_path": cfg.tuning.workspace.path,
                                            "q_dataloader": None,
                                            "benchmark": True})

        adaptor = FRAMEWORKS[framework](framework_specific_info)

        assert cfg.evaluation is not None, 'benchmark need evaluation filed not be None'
        results = {}
        for mode in cfg.evaluation.keys():
            iteration = -1 if deep_get(cfg, 'evaluation.{}.iteration'.format(mode)) is None \
                else deep_get(cfg, 'evaluation.{}.iteration'.format(mode))
            metric =  deep_get(cfg, 'evaluation.{}.metric'.format(mode))
            b_postprocess_cfg = deep_get(cfg, 'evaluation.{}.postprocess'.format(mode))

            if b_dataloader is None:
                assert deep_get(cfg, 'evaluation.{}.dataloader'.format(mode)) is not None, \
                    'dataloader field of yaml file is missing'

                b_dataloader_cfg = deep_get(cfg, 'evaluation.{}.dataloader'.format(mode))
                b_dataloader = create_dataloader(self.framework, b_dataloader_cfg)
                b_func = create_eval_func(self.framework, \
                                          b_dataloader, \
                                          adaptor, \
                                          metric, \
                                          b_postprocess_cfg,
                                          iteration=iteration)
            else:
                b_func = create_eval_func(self.framework, \
                                          b_dataloader, \
                                          adaptor, \
                                          metric, \
                                          b_postprocess_cfg,
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

    def dataloader(self, dataset, batch_size=1, collate_fn=None, last_batch='rollover',
                   sampler=None, batch_sampler=None, num_workers=0, pin_memory=False):
        return DATALOADER(framework=self.framework, dataset=dataset,
                          batch_size=batch_size, collate_fn=collate_fn, last_batch=last_batch,
                          sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
                          pin_memory=pin_memory)

    def metric(self, name, metric_cls, **kwargs):
        metric_cfg = {name : {**kwargs}}
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.metric", metric_cfg)
        metrics = METRICS(self.framework)
        metrics.register(name, metric_cls)

    def postprocess(self, name, postprocess_cls, **kwargs):
        postprocess_cfg = {name : {**kwargs}}
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.postprocess.transform", postprocess_cfg)
        postprocesses = TRANSFORMS(self.framework, 'postprocess')
        postprocesses.register(name, postprocess_cls)
