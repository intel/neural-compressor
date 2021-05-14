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
from ..pruning_modifier import PRUNING_MODIFIERS
from ..utils import logger
from ..utils.utility import singleton
from ..utils.create_obj_from_config import create_train_func
from ..model import BaseModel as LpotModel
from ..adaptor import FRAMEWORKS

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
        self._calib_dataloader = None

    def on_epoch_begin(self, epoch):
        """ called on the begining of epochs"""
        for pruning_modifier in self.pruning_modifiers:
            pruning_modifier.on_epoch_begin(epoch)

    def on_batch_begin(self, batch_id):
        """ called on the begining of batches"""
        for pruning_modifier in self.pruning_modifiers:
            pruning_modifier.on_batch_begin(batch_id)

    def on_batch_end(self):
        """ called on the end of batches"""
        for pruning_modifier in self.pruning_modifiers:
            pruning_modifier.on_batch_end()

    def on_epoch_end(self):
        """ called on the end of epochs"""
        for pruning_modifier in self.pruning_modifiers:
            pruning_modifier.on_epoch_end()
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
                                   'random_seed': self.cfg.tuning.random_seed,
                                   'q_dataloader': None}

        if self.framework == 'tensorflow':
            framework_specific_info.update(
                {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})

        self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)

        assert isinstance(self._model, LpotModel), 'need set lpot Model for quantization....'

        self.pruning_modifiers = []
        for name in self.cfg['pruning']['approach']:
            assert name = 'weight_magnitude', 'now we only support weight_magnitude'
            for magnitude_prune_modifier in self.cfg['pruning']['approach']['weight_magnitude']: 
                self.pruning_modifiers.append(PRUNING_MODIFIERS['MagnitudePruningModifier'])(\
                                        self._model, \
                                        magnitude_prune_modifier, 
                                        self.cfg['pruning']['approach']['weight_magnitude'])
            # TODO, add gradient_sensativity

        if self._calib_dataloader is None and self._calib_func is None:
            calib_dataloader_cfg = deep_get(cfg, 'pruning.train.dataloader')
            assert calib_dataloader_cfg is not None, \
                   'dataloader field of train field of pruning section ' \
                   'in yaml file should be configured as calib_dataloader property is NOT set!'

            self._calib_dataloader = create_dataloader(self.framework, calib_dataloader_cfg) 

        if self._calib_func == None:
            # train section of pruning section in yaml file should be configured.
            train_cfg = deep_get(cfg, 'pruning.train')
            assert train_cfg, "train field of pruning section in yaml file must " \
                              "be configured for pruning if q_func is NOT set."
            hooks = {
                'on_epoch_start': self.on_epoch_begin,
                'on_epoch_end': self.on_epoch_end,
                'on_batch_start': self.on_batch_start,
                'on_batch_end': self.on_batch_end,
            }
            self._calib_func = create_train_func(self.framework, self.calib_dataloader, self.adaptor, train_cfg, hooks=hooks)

        return self._calib_func(self._model.model)

    @property
    def calib_dataloader(self):
        return self._calib_dataloader

    @calib_dataloader.setter
    def calib_dataloader(self, dataloader):
        """Set Data loader for calibration, mandatory for post-training quantization.
           It is iterable and the batched data should consists of a tuple like
           (input, label) if the calibration dataset containing label, or yield (input, _)
           for label-free calibration dataset, the input in the batched data will be used for
           model inference, so it should satisfy the input format of specific model.
           In calibration process, label of data loader will not be used and
           neither the postprocess and metric. User only need to set
           calib_dataloader when calib_dataloader can not be configured from yaml file.

           Args:
               dataloader(generator): user are supported to set a user defined dataloader
                                      which meet the requirements that can yield tuple of
                                      (input, label)/(input, _) batched data. Another good
                                      practice is to use lpot.experimental.common.DataLoader
                                      to initialize a lpot dataloader object. Notice
                                      lpot.experimental.common.DataLoader is just a wrapper of the
                                      information needed to build a dataloader, it can't yield
                                      batched data and only in this setter method
                                      a 'real' calib_dataloader will be created,
                                      the reason is we have to know the framework info
                                      and only after the Quantization object created then
                                      framework infomation can be known.
                                      Future we will support creating iterable dataloader
                                      from lpot.experimental.common.DataLoader
        """
        from .common import _generate_common_dataloader
        self._calib_dataloader = _generate_common_dataloader(
            dataloader, self.framework)

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
        self._calib_func = user_q_func
