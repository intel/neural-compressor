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

import os

from .nas import NASBase
from .nas_utils import nas_registry
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.conf.config import Conf, NASConfig
from neural_compressor.experimental.component import Component
from neural_compressor.utils.create_obj_from_config import \
    create_dataloader, create_train_func, create_eval_func

@nas_registry("Basic")
class BasicNAS(NASBase, Component):
    """

    Args:
        conf_fname (string): The path to the YAML configuration file.
        search_space (dict): A dictionary for defining the search space.
        model_builder (function obj): A function to build model instance with the specified
            model architecture parameters.

    """
    def __init__(self, conf_fname_or_obj, search_space=None, model_builder=None):
        NASBase.__init__(self, search_space=search_space, model_builder=model_builder)
        Component.__init__(self)
        self.init_by_cfg(conf_fname_or_obj)

    def execute(self):
        return self.search()

    def estimate(self, model):
        """Estimate performance of the model.
           Depends on specific NAS algorithm. Here we use train and evaluate.

        Returns:
            Evaluated metrics of the model.
        """
        assert self._train_func is not None and self._eval_func is not None, \
            "train_func and eval_func must be set."
        self._train_func(model)
        return self._eval_func(model)

    def init_by_cfg(self, conf_fname_or_obj):
        if isinstance(conf_fname_or_obj, str):
            if os.path.isfile(conf_fname_or_obj):
                self.conf = Conf(conf_fname_or_obj)
            else: # pragma: no cover
                raise FileNotFoundError(
                    "{} is not a file, please provide a NAS config file path.".format(
                        conf_fname_or_obj
                    )
                )
        elif isinstance(conf_fname_or_obj, NASConfig):
            conf_fname_or_obj.validate()
            self.conf = conf_fname_or_obj
        else: # pragma: no cover
            raise NotImplementedError(
                "Please provide a str path to the config file or an object of NASConfig."
            )
        self._init_with_conf()
        assert self.cfg.nas is not None, "nas section must be set"
        # search related config
        self.init_search_cfg(self.cfg.nas)

    def pre_process(self):
        framework_specific_info = {'device': self.cfg.device,
                                   'random_seed': self.cfg.tuning.random_seed,
                                   'workspace_path': self.cfg.tuning.workspace.path,
                                   'q_dataloader': None}

        if self.framework == 'tensorflow' or self.framework == 'tensorflow_itex':
            framework_specific_info.update(
                {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})

        self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)

        # create dataloaders
        if self._train_dataloader is None and self._train_func is None:
            train_dataloader_cfg = self.cfg.train.dataloader
            assert train_dataloader_cfg is not None, \
                   'No training dataloader setting in current component. Please check ' \
                   'dataloader field of train field in yaml file. Or manually pass ' \
                   'dataloader to component.'

            self._train_dataloader = create_dataloader(self.framework, train_dataloader_cfg)
        if self._eval_dataloader is None and self._eval_func is None:
            eval_dataloader_cfg = self.cfg.evaluation.accuracy.dataloader
            assert eval_dataloader_cfg is not None, \
                'No evaluation dataloader setting in current component. Please check ' \
                'dataloader field of evaluation field in yaml file. Or manually pass ' \
                'dataloader to component.'
            self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)

        # create functions
        if self._train_func is None:
            self._train_func = create_train_func(self.framework,
                                                 self._train_dataloader,
                                                 self.adaptor,
                                                 self.cfg.train,
                                                 hooks=self.hooks)
        if self._eval_func is None:
            metric = [self._metric] if self._metric else self.cfg.evaluation.accuracy.metric
            self._eval_func = create_eval_func(self.framework,
                                               self._eval_dataloader,
                                               self.adaptor,
                                               metric,
                                               self.cfg.evaluation.accuracy.postprocess,
                                               fp32_baseline = False)

    def __repr__(self):
        return 'BasicNAS' # pragma: no cover