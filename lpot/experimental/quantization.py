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
import pickle
import random
import numpy as np
from ..conf.config import Conf
from ..conf.dotdict import deep_get, deep_set, DotDict
from ..strategy import STRATEGIES
from ..utils import logger
from ..utils.utility import time_limit, set_backend
from ..utils.create_obj_from_config import create_dataloader
from .common import Model as LpotModel
from ..model import BaseModel

class Quantization(object):
    """Quantization class automatically searches for optimal quantization recipes for low
       precision model inference, achieving best tuning objectives like inference performance
       within accuracy loss constraints.

       Tuner abstracts out the differences of quantization APIs across various DL frameworks
       and brings a unified API for automatic quantization that works on frameworks including
       tensorflow, pytorch and mxnet.

       Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria
       (<1% or <0.1% etc.) and tuning objectives (performance, memory footprint etc.).
       Tuner class provides a flexible configuration interface via YAML for users to specify
       these parameters.

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal,
        tuning objective and preferred calibration & quantization tuning space etc.

    """

    def __init__(self, conf_fname):
        self.conf = Conf(conf_fname)
        cfg = self.conf.usr_cfg
        self.framework = cfg.model.framework.lower()
        seed = cfg.tuning.random_seed
        random.seed(seed)
        np.random.seed(seed)

        self._model = None
        self._calib_dataloader = None
        self._calib_func = None
        self._eval_dataloader = None
        self._eval_func = None
        set_backend(self.framework)

    def __call__(self):
        """The main entry point of automatic quantization tuning.

           This interface works on all the DL frameworks that lpot supports
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in calibration and evaluation phases
              and quantization tuning settings.

              For this usage, only model parameter is mandatory.

           b) Partial yaml configuration: User specifies dataloaders used in calibration
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create lpot dataloader before calling this
              function.

              After that, User specifies fp32 "model", calibration dataset "calib_dataloader"
              and evaluation dataset "eval_dataloader".
              The calibrated and quantized model is evaluated with "eval_dataloader"
              with evaluation metrics specified in the configuration file. The evaluation tells
              the tuner whether the quantized model meets the accuracy criteria. If not,
              the tuner starts a new calibration and tuning flow.

              For this usage, model, calib_dataloader and eval_dataloader parameters are mandatory.

           c) Partial yaml configuration: User specifies dataloaders used in calibration phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The calibrated and quantized model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the quantized model meets
              the accuracy criteria. If not, the Tuner starts a new calibration and tuning flow.

              For this usage, model, calib_dataloader and eval_func parameters are mandatory.

        Returns:
            quantized model: best qanitized model found, otherwise return None

        """
        cfg = self.conf.usr_cfg

        assert isinstance(self._model, BaseModel), 'need set your Model for quantization....'

        # when eval_func is set, will be directly used and eval_dataloader can be None
        if self._eval_func is None:
            if self._eval_dataloader is None:
                eval_dataloader_cfg = deep_get(cfg, 'evaluation.accuracy.dataloader')
                if eval_dataloader_cfg is None:
                    self._eval_func = self._fake_eval_func
                else:
                    if deep_get(cfg, 'evaluation.accuracy.iteration') == -1 and 'dummy_v2' \
                        in deep_get(cfg, 'evaluation.accuracy.dataloader.dataset', {}):
                        deep_set(cfg, 'evaluation.accuracy.iteration', 10)

                    self._eval_dataloader = create_dataloader(self.framework, \
                                                              eval_dataloader_cfg)

        approach_cfg = deep_get(cfg, 'quantization.approach')
        if self._calib_func:
            assert approach_cfg == 'quant_aware_training', 'q_func property should not ' \
                   'set for {}'.format(approach_cfg)
            assert self._calib_dataloader is None, 'q_func has provided by user, ' \
                   'calib_dataloader property should not be set.'

        if self._calib_dataloader is None and self._calib_func is None:
            if approach_cfg == 'post_training_static_quant':
                calib_dataloader_cfg = deep_get(cfg, 'quantization.calibration.dataloader')
                assert calib_dataloader_cfg is not None, \
                       'dataloader field of calibration field of quantization section ' \
                       'in yaml file should be configured as calib_dataloader property is NOT set!'
                if deep_get(calib_dataloader_cfg, 'shuffle'):
                    logger.warning("post_training_static_quant doesn't support shuffle in "
                                   "dataloader, reset it to False")
                    deep_set(calib_dataloader_cfg, 'shuffle', False)
            elif approach_cfg == 'quant_aware_training':
                calib_dataloader_cfg = deep_get(cfg, 'quantization.train.dataloader')
                assert calib_dataloader_cfg is not None, \
                       'dataloader field of train field of quantization section ' \
                       'in yaml file should be configured as calib_dataloader property is NOT set!'
            else:
                calib_dataloader_cfg = None

            if calib_dataloader_cfg:
                self._calib_dataloader = create_dataloader(self.framework, calib_dataloader_cfg)

        strategy = cfg.tuning.strategy.name.lower()
        assert strategy in STRATEGIES, "Tuning strategy {} is NOT supported".format(strategy)

        _resume = None
        # check if interrupted tuning procedure exists. if yes, it will resume the
        # whole auto tune process.
        self.resume_file = os.path.abspath(os.path.expanduser(cfg.tuning.workspace.resume)) \
                           if cfg.tuning.workspace and cfg.tuning.workspace.resume else None
        if self.resume_file:
            assert os.path.exists(self.resume_file), \
                "The specified resume file {} doesn't exist!".format(self.resume_file)
            with open(self.resume_file, 'rb') as f:
                _resume = pickle.load(f).__dict__

        self.strategy = STRATEGIES[strategy](
            self._model,
            self.conf,
            self._calib_dataloader,
            self._calib_func,
            self._eval_dataloader,
            self._eval_func,
            _resume)

        try:
            with time_limit(self.conf.usr_cfg.tuning.exit_policy.timeout):
                self.strategy.traverse()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error("Unexpected exception {} happened during turing!".format(repr(e)))
        finally:
            if self.strategy.best_qmodel:
                logger.info(
                    "Specified timeout or max trials is reached! "
                    "Found a quantized model which meet accuracy goal. Exit...")
                self.strategy.deploy_config()
            else:
                logger.error(
                    "Specified timeout or max trials is reached! "
                    "Not found any quantized model which meet accuracy goal. Exit...")

            return self.strategy.best_qmodel

    def dataset(self, dataset_type, *args, **kwargs):
        from ..data import DATASETS
        return DATASETS(self.framework)[dataset_type](*args, **kwargs)

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
    def eval_dataloader(self):
        return self._eval_dataloader

    @eval_dataloader.setter
    def eval_dataloader(self, dataloader):
        """Set Data loader for evaluation, It is iterable and the batched data
           should consists of a tuple like (input, label), when eval_dataloader is set,
           user should configure postprocess(optional) and metric in yaml file or set
           postprocess and metric cls. Notice evaluation dataloader will be used to
           generate data for model inference, make sure the input data can be feed to model.

           Args:
               dataloader(generator): user are supported to set a user defined dataloader
                                      which meet the requirements that can yield tuple of
                                      (input, label)/(input, _) batched data.
                                      Another good practice is to use
                                      lpot.experimental.common.DataLoader
                                      to initialize a lpot dataloader object.
                                      Notice lpot.experimental.common.DataLoader
                                      is just a wrapper of the information needed to
                                      build a dataloader, it can't yield
                                      batched data and only in this setter method
                                      a 'real' eval_dataloader will be created,
                                      the reason is we have to know the framework info
                                      and only after the Quantization object created then
                                      framework infomation can be known.
                                      Future we will support creating iterable dataloader
                                      from lpot.experimental.common.DataLoader

        """
        from .common import _generate_common_dataloader
        self._eval_dataloader = _generate_common_dataloader(
            dataloader, self.framework)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object

        Args:
           user_model: user are supported to set model from original framework model format
                       (eg, tensorflow frozen_pb or path to a saved model),
                       but not recommended. Best practice is to set from a initialized
                       lpot.experimental.common.Model.
                       If tensorflow model is used, model's inputs/outputs will be
                       auto inferenced, but sometimes auto inferenced
                       inputs/outputs will not meet your requests,
                       set them manually in config yaml file.
                       Another corner case is slim model of tensorflow,
                       be careful of the name of model configured in yaml file,
                       make sure the name is in supported slim model list.

        """
        if not isinstance(user_model, BaseModel):
            logger.warning('force convert user raw model to lpot model, ' +
                'better initialize lpot.experimental.common.Model and set....')
            self._model = LpotModel(user_model)
        else:
            self._model = user_model

        cfg = self.conf.usr_cfg
        if self.framework == 'tensorflow':
            self._model.name = cfg.model.name
            # (TODO) ugly tensorflow should use outputs before inputs on checkpoint case
            self._model.output_tensor_names = cfg.model.outputs
            self._model.input_tensor_names = cfg.model.inputs
            self._model.workspace_path = cfg.tuning.workspace.path


    @property
    def metric(self):
        logger.warning('metric not support getter....')
        return None

    @metric.setter
    def metric(self, user_metric):
        """Set metric class and lpot will initialize this class when evaluation
           lpot have many built-in metrics, but user can set specific metric through
           this api. The metric class should take the outputs of the model or
           postprocess(if have) as inputs, lpot built-in metric always take
           (predictions, labels) as inputs for update,
           and user_metric.metric_cls should be sub_class of lpot.metric.BaseMetric.

        Args:
            user_metric(lpot.experimental.common.Metric):
                user_metric should be object initialized from
                lpot.experimental.common.Metric, in this method the
                user_metric.metric_cls will be registered to
                specific frameworks and initialized.

        """
        from .common import Metric as LpotMetric
        assert isinstance(user_metric, LpotMetric), \
            'please initialize a lpot.experimental.common.Metric and set....'

        metric_cfg = {user_metric.name : {**user_metric.kwargs}}
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.metric"):
            logger.warning('already set metric in yaml file, will override it...')
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.metric", metric_cfg)
        self.conf.usr_cfg = DotDict(self.conf.usr_cfg)
        from ..metric import METRICS
        metrics = METRICS(self.framework)
        metrics.register(user_metric.name, user_metric.metric_cls)

    @property
    def postprocess(self, user_postprocess):
        logger.warning('postprocess not support getter....')
        return None

    @postprocess.setter
    def postprocess(self, user_postprocess):
        """Set postprocess class and lpot will initialize this class when evaluation.
           The postprocess class should take the outputs of the model as inputs, and
           output (predictions, labels) as inputs for metric update.
           user_postprocess.postprocess_cls should be sub_class of lpot.data.BaseTransform.

        Args:
            user_postprocess(lpot.experimental.common.Postprocess):
                user_postprocess should be object initialized from
                lpot.experimental.common.Postprocess,
                in this method the user_postprocess.postprocess_cls will be
                registered to specific frameworks and initialized.

        """
        from .common import Postprocess as LpotPostprocess
        assert isinstance(user_postprocess, LpotPostprocess), \
            'please initialize a lpot.experimental.common.Postprocess and set....'
        postprocess_cfg = {user_postprocess.name : {**user_postprocess.kwargs}}
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.postprocess"):
            logger.warning('already set postprocess in yaml file, will override it...')
        deep_set(
            self.conf.usr_cfg, "evaluation.accuracy.postprocess.transform", postprocess_cfg)
        from ..data import TRANSFORMS
        postprocesses = TRANSFORMS(self.framework, 'postprocess')
        postprocesses.register(user_postprocess.name, user_postprocess.postprocess_cls)
        logger.info("{} registered to postprocess".format(user_postprocess.name))

    # if user doesn't config evaluation dataloader in yaml and eval_func is None, a
    # fake eval func is created to do quantization once without tuning
    def _fake_eval_func(self, model):
        return 1.

    # BELOW API TO BE DEPRECATED!
    @property
    def q_func(self):
        logger.warning('q_func not support getter....')
        return None

    @q_func.setter
    def q_func(self, user_q_func):
        """Training function for Quantization-Aware Training.
           It is optional and only takes effect when user choose
           "quant_aware_training" approach in yaml.

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

    @property
    def eval_func(self):
        logger.warning('eval_func not support getter....')
        return None

    @eval_func.setter
    def eval_func(self, user_eval_func):
        """ The evaluation function provided by user.

        Args:
            user_eval_func: This function takes model as parameter,
                            and evaluation dataset and metrics should be
                            encapsulated in this function implementation
                            and outputs a higher-is-better accuracy scalar
                            value.

                            The pseudo code should be something like:

                            def eval_func(model):
                                 input, label = dataloader()
                                 output = model(input)
                                 accuracy = metric(output, label)
                                 return accuracy
        """
        self._eval_func = user_eval_func

    def __repr__(self):
        return 'Quantization'
