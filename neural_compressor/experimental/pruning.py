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

from .component import Component
from ..pruners import PRUNERS
from ..utils import logger
from ..utils.utility import singleton, time_limit, set_backend
from ..utils.create_obj_from_config import create_dataloader, create_train_func, create_eval_func
from ..model import BaseModel
from .common import Model
from ..adaptor import FRAMEWORKS
from ..conf.config import Pruning_Conf

class Pruning(Component):
    """This is base class of pruning object.

       Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria
       (<1% or <0.1% etc.) and pruning objectives (performance, memory footprint etc.).
       Pruning class provides a flexible configuration interface via YAML for users to specify
       these parameters.

    Args:
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or
            Pruning_Conf class containing accuracy goal, pruning objective and related
            dataloaders etc.

    """

    def __init__(self, conf_fname_or_obj=None):
        super(Pruning, self).__init__()
        if isinstance(conf_fname_or_obj, Pruning_Conf):
            self.conf = conf_fname_or_obj
        else:
            self.conf = Pruning_Conf(conf_fname_or_obj)
        self._init_with_conf()

        self._pruning_func = None
        self.pruners = []

    def _pre_epoch_begin(self):
        """ called before training """
        for pruner in self.pruners:
            pruner.pre_epoch_begin()

    def _on_epoch_begin(self, epoch):
        """ called on the beginning of epochs"""
        for pruner in self.pruners:
            pruner.on_epoch_begin(epoch)

    def _on_batch_begin(self, batch_id):
        """ called on the beginning of batches"""
        res = []
        for pruner in self.pruners:
            res.append(pruner.on_batch_begin(batch_id))

        return res

    def _on_post_grad(self):
        """ called after gradient computed """
        """ usually for getting gradients """
        for pruner in self.pruners:
            pruner.on_post_grad()

    def _on_batch_end(self):
        """ called on the end of batches"""
        res = []
        for pruner in self.pruners:
            res.append(pruner.on_batch_end())
        return res

    def _on_epoch_end(self):
        """ called on the end of epochs"""
        res = []
        for pruner in self.pruners:
            res.append(pruner.on_epoch_end())
        stats, sparsity = self._model.report_sparsity()
        logger.info(stats)
        logger.info(sparsity)
        return res

    def _post_epoch_end(self):
        """ called after training """
        for pruner in self.pruners:
            pruner.post_epoch_end()

    def pre_process(self):
        assert isinstance(self._model, BaseModel), 'need set neural_compressor Model for pruning....'

        framework_specific_info = {'device': self.cfg.device,
                                   'random_seed': self.cfg.tuning.random_seed,
                                   'workspace_path': self.cfg.tuning.workspace.path,
                                   'q_dataloader': None}

        if self.framework == 'tensorflow':
            framework_specific_info.update(
                {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})

        self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)

        self.generate_hooks()
        self.generate_pruners()

        if self._train_dataloader is None and self._pruning_func is None:
            train_dataloader_cfg = self.cfg.pruning.train.dataloader
            assert train_dataloader_cfg is not None, \
                   'dataloader field of train field of pruning section ' \
                   'in yaml file should be configured as train_dataloader property is NOT set!'

            self._train_dataloader = create_dataloader(self.framework, train_dataloader_cfg)

        if self._eval_dataloader is None and self._eval_func is None:
            eval_dataloader_cfg = self.cfg.evaluation.accuracy.dataloader
            assert eval_dataloader_cfg is not None, \
                   'dataloader field of evaluation ' \
                   'in yaml file should be configured as eval_dataloader property is NOT set!'

            self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)

        if self._pruning_func is None:
            # train section of pruning section in yaml file should be configured.
            train_cfg = self.cfg.pruning.train
            assert train_cfg, "train field of pruning section in yaml file must " \
                              "be configured for pruning if pruning_func is NOT set."
            self._pruning_func = create_train_func(self.framework, \
                                                   self.train_dataloader, \
                                                   self.adaptor, train_cfg, hooks=self.hooks)
        if self._eval_func is None:
            # eval section in yaml file should be configured.
            eval_cfg = self.cfg.evaluation
            assert eval_cfg, "eval field of pruning section in yaml file must " \
                              "be configured for pruning if eval_func is NOT set."
            self._eval_func = create_eval_func(self.framework, \
                                               self.eval_dataloader, \
                                               self.adaptor, \
                                               eval_cfg.accuracy.metric, \
                                               eval_cfg.accuracy.postprocess, \
                                               fp32_baseline = False)
        if getattr(self.train_dataloader, 'distributed', False):
            self.register_hook('pre_epoch_begin', self.adaptor._pre_hook_for_hvd)

    def execute(self):
        self._pruning_func(self._model if getattr(self._pruning_func, 'builtin', None) \
                        else self._model.model)
        logger.info("Model pruning is done. Start to evaluate the pruned model.")
        score = self._eval_func(self._model if getattr(self._eval_func, 'builtin', None) \
                        else self._model.model)
        logger.info("Pruned model score is {}.".format(str(score)))

        return self._model

    def generate_hooks(self):
        # register hooks for pruning
        self.register_hook('pre_epoch_begin', self._pre_epoch_begin)
        self.register_hook('post_epoch_end', self._post_epoch_end)
        self.register_hook('on_epoch_begin', self._on_epoch_begin)
        self.register_hook('on_epoch_end', self._on_epoch_end)
        self.register_hook('on_batch_begin', self._on_batch_begin)
        self.register_hook('on_batch_end', self._on_batch_end)
        self.register_hook('on_post_grad', self._on_post_grad)

    def generate_pruners(self):
        for name in self.cfg.pruning.approach:
            assert name == 'weight_compression', 'now we only support weight_compression'
            for pruner in self.cfg.pruning.approach.weight_compression.pruners:
                if pruner.prune_type == 'basic_magnitude':
                    self.pruners.append(PRUNERS['BasicMagnitude'](\
                                            self._model, \
                                            pruner,
                                            self.cfg.pruning.approach.weight_compression))
                if pruner.prune_type == 'pattern_lock':
                    self.pruners.append(PRUNERS['PatternLock'](\
                                            self._model, \
                                            pruner,
                                            self.cfg.pruning.approach.weight_compression))
                elif pruner.prune_type == 'gradient_sensitivity':
                    self.pruners.append(PRUNERS['GradientSensitivity'](\
                                            self._model, \
                                            pruner,
                                            self.cfg.pruning.approach.weight_compression))

    def __call__(self):
        """The main entry point of pruning.

           This interface currently only works on pytorch
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in training and evaluation phases
              and pruning tuning settings.

              For this usage, only model parameter is mandatory.

           b) Partial yaml configuration: User specifies dataloaders used in training
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create neural_compressor dataloader before calling this
              function.

              After that, User specifies fp32 "model", training dataset "p_dataloader"
              and evaluation dataset "eval_dataloader".

              For this usage, model, p_dataloader and eval_dataloader parameters are mandatory.

           c) Partial yaml configuration: User specifies dataloaders used in training phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The trained and pruned model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the pruned model meets
              the accuracy criteria. If not, the Tuner starts a new training and tuning flow.

              For this usage, model, q_dataloader and eval_func parameters are mandatory.

        Returns:
            pruned model: best pruned model found, otherwise return None

        """
        return super(Pruning, self).__call__()

    @property
    def pruning_func(self):
        """ not support get pruning_func """
        assert False, 'Should not try to get the value of `pruning_func` attribute.'
        return None

    @pruning_func.setter
    def pruning_func(self, user_pruning_func):
        """Training function for pruning.

        Args:
            user_pruning_func: This function takes "model" as input parameter
                         and executes entire training process with self
                         contained training hyper-parameters. If pruning_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func
                         to make evaluation of the model executed.
        """
        self._pruning_func = user_pruning_func

    def __repr__(self):
        return 'Pruning'
