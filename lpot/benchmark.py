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

from .adaptor import FRAMEWORKS
from .objective import OBJECTIVES
from .conf.config import Conf
from .utils import logger
from .utils.create_obj_from_config import create_eval_func, create_dataloader
from .conf.dotdict import deep_get, deep_set
from .model import BaseModel as LpotModel

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
        self._model = None
        self._b_dataloader = None

    def __call__(self):
        cfg = self.conf.usr_cfg
        framework_specific_info = {'device': cfg.device, \
                                   'approach': cfg.quantization.approach, \
                                   'random_seed': cfg.tuning.random_seed}
        framework = cfg.model.framework.lower()
        if framework == 'tensorflow':
            framework_specific_info.update({"inputs": cfg.model.inputs, \
                                            "outputs": cfg.model.outputs, \
                                            "recipes": cfg.model.recipes, \
                                            'workspace_path': cfg.tuning.workspace.path})
        if framework == 'mxnet':
            framework_specific_info.update({"b_dataloader": self._b_dataloader})
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

        assert isinstance(self._model, LpotModel), 'need set lpot Model for quantization....'

        adaptor = FRAMEWORKS[framework](framework_specific_info)

        assert cfg.evaluation is not None, 'benchmark need evaluation filed not be None'
        results = {}
        for mode in cfg.evaluation.keys():
            iteration = -1 if deep_get(cfg, 'evaluation.{}.iteration'.format(mode)) is None \
                else deep_get(cfg, 'evaluation.{}.iteration'.format(mode))
            metric =  deep_get(cfg, 'evaluation.{}.metric'.format(mode))
            b_postprocess_cfg = deep_get(cfg, 'evaluation.{}.postprocess'.format(mode))

            if self._b_dataloader is None:
                assert deep_get(cfg, 'evaluation.{}.dataloader'.format(mode)) is not None, \
                    'dataloader field of yaml file is missing'

                b_dataloader_cfg = deep_get(cfg, 'evaluation.{}.dataloader'.format(mode))
                self._b_dataloader = create_dataloader(self.framework, b_dataloader_cfg)
                b_func = create_eval_func(self.framework, \
                                          self._b_dataloader, \
                                          adaptor, \
                                          metric, \
                                          b_postprocess_cfg,
                                          iteration=iteration)
            else:
                b_func = create_eval_func(self.framework, \
                                          self._b_dataloader, \
                                          adaptor, \
                                          metric, \
                                          b_postprocess_cfg,
                                          iteration=iteration)

            objective = cfg.tuning.objective.lower()
            self.objective = OBJECTIVES[objective](cfg.tuning.accuracy_criterion, \
                                                   is_measure=True)

            val = self.objective.evaluate(b_func, self._model)
            logger.info('{} mode benchmark done!'.format(mode))
            # measurer contain info not only performance(eg, memory, model_size)
            # also measurer have result list among steps
            acc, _ = val
            batch_size = self._b_dataloader.batch_size
            warmup =  0 if deep_get(cfg, 'evaluation.{}.warmup'.format(mode)) is None \
                else deep_get(cfg, 'evaluation.{}.warmup'.format(mode))

            assert len(self.objective.measurer.result_list()) > warmup, \
                'itreation should larger than warmup'

            results[mode] = acc, batch_size, \
                            self.objective.measurer.result_list()[warmup:]

        return results

    @property
    def b_dataloader(self):
        return self._b_dataloader

    @b_dataloader.setter
    def b_dataloader(self, dataloader):
        from .common import _generate_common_dataloader
        self._b_dataloader = _generate_common_dataloader(dataloader, self.framework)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, user_model):
        from .common import Model as LpotModel
        from .model import MODELS
        if not isinstance(user_model, LpotModel):
            logger.warning('force convert user raw model to lpot model, \
                better initialize lpot.common.Model and set....')
            user_model = LpotModel(user_model)

        framework_model_info = {}
        cfg = self.conf.usr_cfg
        if self.framework == 'tensorflow':
            framework_model_info.update(
                {'name': cfg.model.name,
                 'input_tensor_names': cfg.model.inputs,
                 'output_tensor_names': cfg.model.outputs,
                 'workspace_path': cfg.tuning.workspace.path})

        self._model = MODELS[self.framework](\
            user_model.root, framework_model_info, **user_model.kwargs)

    @property
    def metric(self):
        logger.warning('metric not support getter....')
        return None

    @metric.setter
    def metric(self, user_metric):
        from .common import Metric as LpotMetric
        assert isinstance(user_metric, LpotMetric), \
            'please initialize a lpot.common.Metric and set....'

        metric_cfg = {user_metric.name : {**user_metric.kwargs}}
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.metric"):
            logger.warning('already set metric in yaml file, will override it...')
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.metric", metric_cfg)
        from .conf.dotdict import DotDict
        self.conf.usr_cfg = DotDict(self.conf.usr_cfg)
        from .metric import METRICS
        metrics = METRICS(self.framework)
        metrics.register(user_metric.name, user_metric.metric_cls)

    @property
    def postprocess(self, user_postprocess):
        logger.warning('postprocess not support getter....')
        return None

    @postprocess.setter
    def postprocess(self, user_postprocess):
        from .common import Postprocess as LpotPostprocess
        assert isinstance(user_postprocess, LpotPostprocess), \
            'please initialize a lpot.common.Postprocess and set....'
        postprocess_cfg = {user_postprocess.name : {**user_postprocess.kwargs}}
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.postprocess"):
            logger.warning('already set postprocess in yaml file, will override it...')
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.postprocess.transform", postprocess_cfg)
        from .data import TRANSFORMS
        postprocesses = TRANSFORMS(self.framework, 'postprocess')
        postprocesses.register(user_postprocess.name, user_postprocess.postprocess_cls)
        logger.info("{} registered to postprocess".format(user_postprocess.name))

