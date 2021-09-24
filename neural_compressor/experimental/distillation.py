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

import copy
from .component import Component
from ..utils import logger
from ..utils.create_obj_from_config import create_dataloader, \
    create_train_func_for_distillation, create_eval_func
from ..model import BaseModel
from .common import Model
from ..adaptor import FRAMEWORKS
from neural_compressor.experimental.common import Criterions, Optimizers
from ..conf.config import Distillation_Conf

class Distillation(Component):
    """

    Args:
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or
            Distillation_Conf containing accuracy goal, distillation objective and related
            dataloaders etc.

    """

    def __init__(self, conf_fname_or_obj=None):
        super(Distillation, self).__init__()
        if isinstance(conf_fname_or_obj, Distillation_Conf):
            self.conf = conf_fname_or_obj
        else:
            self.conf = Distillation_Conf(conf_fname_or_obj)
        self._init_with_conf()

        self._teacher_model = None
        self._criterion = None
        self._optimizer = None
        self._epoch_runned = 0
        self.eval_frequency = 1
        self.best_score = 0
        self.best_model = None

    def _pre_epoch_begin(self):
        """ called before training """
        assert self.teacher_model, 'teacher_model must be set.'
        score = self._eval_func(self.teacher_model \
                if getattr(self._eval_func, 'builtin', None) else self.teacher_model.model)
        logger.info("teacher model score is {}.".format(str(score)))

        assert self._model, 'student_model must be set.'
        score = self._eval_func(self._model \
                if getattr(self._eval_func, 'builtin', None) else self._model.model)
        logger.info("initial model score is {}.".format(str(score)))

        if self.eval_frequency > 0:
            self.best_score = score
            self.best_model = copy.deepcopy(self._model)

    def on_post_forward(self, input, teacher_output=None):
        """ called after model forward """
        assert self.criterion and hasattr(self.criterion, "teacher_model_forward"), \
            'criterion must be set and have teacher_model_forward func.'
        if teacher_output is not None:
            self.criterion.teacher_outputs = teacher_output
        else:
            self.criterion.teacher_model_forward(input, teacher_model=self.teacher_model.model)

    def _on_epoch_end(self):
        """ called on the end of epochs"""
        self._epoch_runned += 1
        if self.eval_frequency > 0 and \
           self._epoch_runned % self.eval_frequency == 0:
            score = self._eval_func(self._model \
                    if getattr(self._eval_func, 'builtin', None) else self._model.model)
            logger.info("model score of epoch {} is {}.".format(self._epoch_runned, str(score)))
            if score > self.best_score:
                self.best_model = copy.deepcopy(self._model)
                self.best_score = score

    def pre_process(self):
        framework_specific_info = {'device': self.cfg.device,
                                   'random_seed': self.cfg.tuning.random_seed,
                                   'workspace_path': self.cfg.tuning.workspace.path,
                                   'q_dataloader': None}

        if self.framework == 'tensorflow':
            framework_specific_info.update(
                {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})

        self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)

        self.generate_hooks()
        assert isinstance(self._model, BaseModel), 'need set neural_compressor Model for distillation....'

        if self._train_dataloader is None and self._train_func is None:
            train_dataloader_cfg = self.cfg.distillation.train.dataloader
            assert train_dataloader_cfg is not None, \
                   'dataloader field of train field of distillation section ' \
                   'in yaml file should be configured as train_dataloader property is NOT set!'

            self._train_dataloader = create_dataloader(self.framework, train_dataloader_cfg)

        if self._eval_dataloader is None and self._eval_func is None:
            eval_dataloader_cfg = self.cfg.evaluation.accuracy.dataloader
            assert eval_dataloader_cfg is not None, \
                   'dataloader field of evaluation ' \
                   'in yaml file should be configured as eval_dataloader property is NOT set!'

            self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)

        if self._train_func is None:
            # train section of distillation section in yaml file should be configured.
            train_cfg = self.cfg.distillation.train
            assert train_cfg, "train field of distillation section in yaml file must " \
                              "be configured for distillation if train_func is NOT set."
                
            if self.criterion is None:
                assert 'criterion' in train_cfg.keys(), \
                    "criterion part in train field of distillation section in yaml file " \
                    "must be configured for distillation if criterion is NOT set."
                criterion_cfg = train_cfg.criterion
                assert len(criterion_cfg) == 1, "There must be exactly one loss in " \
                    "criterion part, instead got {} loss.".format(len(criterion_cfg))
                loss = list(criterion_cfg.keys())[0]
                loss_cfg = criterion_cfg[loss]
                criterion_builder = Criterions(self.framework)[loss](loss_cfg)
                criterion_tuple = criterion_builder()
                self.criterion = criterion_tuple[0](**criterion_tuple[1])
            else:
                logger.warning("Use user defined criterion, " \
                               "ignoring the criterion setting in yaml file.")

            if self.optimizer is None:
                assert 'optimizer' in train_cfg.keys(), \
                    "optimizer part in train field of distillation section in yaml file " \
                    "must be configured for distillation if optimizer is NOT set."
                optimizer_cfg = train_cfg.optimizer
                assert len(optimizer_cfg) == 1, "There must be exactly one optimizer in " \
                    "optimizer part, instead got {} optimizer.".format(len(optimizer_cfg))
                optimizer_name = list(optimizer_cfg.keys())[0]
                optimizer_cfg_ = optimizer_cfg[optimizer_name]
                optimizer_builder = Optimizers(self.framework)[optimizer_name](optimizer_cfg_)
                optimizer_tuple = optimizer_builder()
                self.optimizer = optimizer_tuple[0](self.model.model.parameters(), \
                                                    **optimizer_tuple[1])
            else:
                logger.warning("Use user defined optimizer, " \
                               "ignoring the optimizer setting in yaml file.")

            assert self.teacher_model, "teacher_model must be set."
            self.criterion.teacher_model = self.teacher_model.model
            train_cfg.criterion = self.criterion
            train_cfg.optimizer = self.optimizer

            self._train_func = create_train_func_for_distillation(self.framework, \
                                                                  self.train_dataloader, \
                                                                  self.adaptor, \
                                                                  train_cfg, \
                                                                  hooks=self.hooks)
        if self._eval_func is None:
            # eval section in yaml file should be configured.
            eval_cfg = self.cfg.evaluation
            assert eval_cfg, "eval field of distillation section in yaml file must " \
                              "be configured for distillation if eval_func is NOT set."
            self._eval_func = create_eval_func(self.framework, \
                                               self.eval_dataloader, \
                                               self.adaptor, \
                                               eval_cfg.accuracy.metric, \
                                               eval_cfg.accuracy.postprocess, \
                                               fp32_baseline = False)

    def execute(self):
        self._train_func(self._model \
                if getattr(self._train_func, 'builtin', None) else self._model.model)
        logger.info("Model distillation is done. Start to evaluate the distilled model.")
        self._model = self.best_model if self.best_model else self._model
        score = self._eval_func(self._model \
                if getattr(self._eval_func, 'builtin', None) else self._model.model)

        logger.info("distilled model score is {}.".format(str(score)))
        return self._model

    def generate_hooks(self):
        # register hooks for distillation
        self.register_hook('pre_epoch_begin', self._pre_epoch_begin)
        self.register_hook('on_epoch_end', self._on_epoch_end)

    def __call__(self):
        """The main entry point of distillation.

           This interface currently only works on pytorch
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in training and evaluation phases
              and distillation settings.

              For this usage, only student_model and teacher_model parameter is mandatory.

           b) Partial yaml configuration: User specifies dataloaders used in training
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create neural_compressor dataloader before calling this
              function.

              After that, User specifies fp32 "model", training dataset "train_dataloader"
              and evaluation dataset "eval_dataloader".

              For this usage, student_model, teacher_model, train_dataloader and eval_dataloader 
              parameters are mandatory.

           c) Partial yaml configuration: User specifies dataloaders used in training phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The trained and distilled model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the distilled model meets
              the accuracy criteria. If not, the Tuner starts a new training and tuning flow.

              For this usage, student_model, teacher_model, train_dataloader and eval_func 
              parameters are mandatory.

        Returns:
            distilled model: best distilled model found, otherwise return None

        """
        return super(Distillation, self).__call__()

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, user_criterion):
        self._criterion = user_criterion

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, user_optimizer):
        self._optimizer = user_optimizer

    @property
    def teacher_model(self):
        """ Getter of model in neural_compressor.model  """
        return self._teacher_model

    @teacher_model.setter
    def teacher_model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object

        Args:
           user_model: user are supported to set model from original framework model format
                       (eg, tensorflow frozen_pb or path to a saved model),
                       but not recommended. Best practice is to set from a initialized
                       neural_compressor.experimental.common.Model.
                       If tensorflow model is used, model's inputs/outputs will be
                       auto inferenced, but sometimes auto inferenced
                       inputs/outputs will not meet your requests,
                       set them manually in config yaml file.
                       Another corner case is slim model of tensorflow,
                       be careful of the name of model configured in yaml file,
                       make sure the name is in supported slim model list.

        """
        if not isinstance(user_model, BaseModel):
            logger.warning("Force convert framework model to neural_compressor model.")
            self._teacher_model = Model(user_model)
        else:
            self._teacher_model = user_model

    @property
    def student_model(self):
        """ Getter of model in neural_compressor.model  """
        return self._model

    @student_model.setter
    def student_model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object

        Args:
           user_model: user are supported to set model from original framework model format
                       (eg, tensorflow frozen_pb or path to a saved model),
                       but not recommended. Best practice is to set from a initialized
                       neural_compressor.experimental.common.Model.
                       If tensorflow model is used, model's inputs/outputs will be
                       auto inferenced, but sometimes auto inferenced
                       inputs/outputs will not meet your requests,
                       set them manually in config yaml file.
                       Another corner case is slim model of tensorflow,
                       be careful of the name of model configured in yaml file,
                       make sure the name is in supported slim model list.

        """
        if not isinstance(user_model, BaseModel):
            logger.warning("Force convert framework model to neural_compressor model.")
            self._model = Model(user_model)
        else:
            self._model = user_model

    def __repr__(self):
        return 'Distillation'
