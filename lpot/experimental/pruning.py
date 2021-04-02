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


from ..conf.config import Conf
from ..policy import POLICIES
from ..utils import logger
from ..utils.utility import singleton
from ..model import BaseModel as LpotModel

@singleton
class Pruning(object):
    """This is base class of pruning object.

       Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria
       (<1% or <0.1% etc.) and pruning objectives (performance, memory footprint etc.).
       Pruning class provides a flexible configuration interface via YAML for users to specify
       these parameters.

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal,
        pruning objective and related dataloaders etc.

    """

    def __init__(self, conf_fname):
        self.conf = Conf(conf_fname)
        self.cfg = self.conf.usr_cfg
        self.framework = self.cfg.model.framework.lower()
        self._model = None
        self._calib_func = None

    def on_epoch_begin(self, epoch):
        """ called on the begining of epochs"""
        for policy in self.policies:
            policy.on_epoch_begin(epoch)

    def on_batch_begin(self, batch_id):
        """ called on the begining of batches"""
        for policy in self.policies:
            policy.on_batch_begin(batch_id)

    def on_batch_end(self):
        """ called on the end of batches"""
        for policy in self.policies:
            policy.on_batch_end()

    def on_epoch_end(self):
        """ called on the end of epochs"""
        for policy in self.policies:
            policy.on_epoch_end()
        stats, sparsity = self._model.report_sparsity()
        logger.info(stats)
        logger.info(sparsity)

    def __call__(self):
        """The main entry point of pruning.

           This interface currently only works on pytorch
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in calibration and evaluation phases
              and quantization tuning settings.

              For this usage, only model parameter is mandotory.

           b) Partial yaml configuration: User specifies dataloaders used in calibration
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create lpot dataloader before calling this
              function.

              After that, User specifies fp32 "model", calibration dataset "q_dataloader"
              and evaluation dataset "eval_dataloader".
              The calibrated and quantized model is evaluated with "eval_dataloader"
              with evaluation metrics specified in the configuration file. The evaluation tells
              the tuner whether the quantized model meets the accuracy criteria. If not,
              the tuner starts a new calibration and tuning flow.

              For this usage, model, q_dataloader and eval_dataloader parameters are mandotory.

           c) Partial yaml configuration: User specifies dataloaders used in calibration phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The calibrated and quantized model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the quantized model meets
              the accuracy criteria. If not, the Tuner starts a new calibration and tuning flow.

              For this usage, model, q_dataloader and eval_func parameters are mandotory.

        Returns:
            pruned model: best pruned model found, otherwise return None

        """
        framework_specific_info = {'device': self.cfg.device,
                                   'approach': self.cfg.quantization.approach,
                                   'random_seed': self.cfg.tuning.random_seed,
                                   'q_dataloader': None}
        if self.framework == 'tensorflow':
            framework_specific_info.update(
                {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})

        assert isinstance(self._model, LpotModel), 'need set lpot Model for quantization....'

        policies = {}
        for policy in POLICIES:
            for name in self.cfg["pruning"][policy]:
                policies[name] = {"policy_name": policy,
                                  "policy_spec": self.cfg["pruning"][policy][name]}
        self.policies = []
        for name, policy_spec in policies.items():
            print(policy_spec)
            self.policies.append(POLICIES[policy_spec["policy_name"]](
                self._model, policy_spec["policy_spec"], self.cfg))
        return self._calib_func(self._model.model)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, user_model):
        """Only support PyTorch model, it's torch.nn.model instance.

        Args:
           user_model: user are supported to set model from original PyTorch model format
                       Best practice is to set from a initialized lpot.experimental.common.Model.

        """
        from .common import Model as LpotModel
        if not isinstance(user_model, LpotModel):
            logger.warning('force convert user raw model to lpot model, ' +
                'better initialize lpot.experimental.common.Model and set....')
            user_model = LpotModel(user_model)
        framework_model_info = {}
        cfg = self.conf.usr_cfg
        if self.framework == 'tensorflow':
            framework_model_info.update(
                {'name': cfg.model.name,
                 'input_tensor_names': cfg.model.inputs,
                 'output_tensor_names': cfg.model.outputs,
                 'workspace_path': cfg.tuning.workspace.path})

        from ..model import MODELS
        self._model = MODELS[self.framework](\
            user_model.root, framework_model_info, **user_model.kwargs)

    @property
    def q_func(self):
        logger.warning('q_func not support getter....')
        return None
        
    @q_func.setter
    def q_func(self, user_q_func):
        """Training function for pruning. 

        Args:
            user_q_func: This function takes "model" as input parameter
                         and executes entire training process with self
                         contained training hyper-parameters. If q_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func 
                         to make evaluation of the model executed.
        """
        logger.warning('q_func is to be deprecated, please construct q_dataloader....')
        self._calib_func = user_q_func
