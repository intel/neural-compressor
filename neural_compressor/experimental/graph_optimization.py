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
import tempfile
import sys
import numpy as np
import yaml
from ..conf.config import Graph_Optimization_Conf
from ..conf.dotdict import deep_get, deep_set, DotDict
from ..strategy import STRATEGIES
from ..utils import logger
from ..utils.create_obj_from_config import create_dataloader
from ..utils.utility import CpuInfo, time_limit, set_backend
from .common import Model as NCModel
from ..model import BaseModel

class Graph_Optimization():
    """Graph_Optimization class automatically searches for optimal quantization recipes for low
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
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or
            Graph_Optimization_Conf class containing accuracy goal, tuning objective and
            preferred calibration & quantization tuning space etc.

    """

    def __init__(self, conf_fname_or_obj=None):
        self.conf_name = conf_fname_or_obj
        self._model = None
        self._eval_dataloader = None
        self._eval_func = None
        self._precisions = 'fp32'
        self._input = []
        self._output = []
        self.conf = None

        if isinstance(conf_fname_or_obj, Graph_Optimization_Conf):
            self.conf = conf_fname_or_obj
        else:
            self.conf = Graph_Optimization_Conf(conf_fname_or_obj)
        cfg = self.conf.usr_cfg
        if cfg.model.framework != 'NA':
            self.framework = cfg.model.framework.lower()
            set_backend(self.framework)

        cfg.tuning.strategy.name = 'automixedprecision'
        seed = cfg.tuning.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def __call__(self):
        """The main entry point of graph optimization process.

           This interface works on all the DL frameworks that neural_compressor supports
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in calibration and evaluation phases
              and quantization tuning settings.

              For this usage, only model parameter is mandatory.

           b) Partial yaml configuration: User specifies dataloaders used in calibration
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create neural_compressor dataloader before calling this
              function.

              After that, User specifies fp32 "model", calibration dataset "calib_dataloader"
              and evaluation dataset "eval_dataloader".
              The calibrated and converted model is evaluated with "eval_dataloader"
              with evaluation metrics specified in the configuration file. The evaluation tells
              the tuner whether the converted model meets the accuracy criteria. If not,
              the tuner starts a new calibration and tuning flow.

              For this usage, model, calib_dataloader and eval_dataloader parameters are mandatory.

           c) Partial yaml configuration: User specifies dataloaders used in calibration phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The calibrated and converted model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the converted model meets
              the accuracy criteria. If not, the Tuner starts a new calibration and tuning flow.

              For this usage, model, calib_dataloader and eval_func parameters are mandatory.

        Returns:
            converted model: best converted model found, otherwise return None

        """

        assert isinstance(self._model, BaseModel), 'need set your Model for quantization....'

        cfg = self.conf.usr_cfg
        if self.framework == 'tensorflow':
            self._model.name = cfg.model.name
            self._model.output_tensor_names = cfg.model.outputs
            self._model.input_tensor_names = cfg.model.inputs
            self._model.workspace_path = cfg.tuning.workspace.path
        else: 
            logger.warning("Only TensorFlow graph optimization is supported at current stage.") 
            sys.exit(0) 

        # when eval_func is set, will be directly used and eval_dataloader can be None
        if self._eval_func is None:
            if self._eval_dataloader is None:
                eval_dataloader_cfg = deep_get(cfg, 'evaluation.accuracy.dataloader')
                if eval_dataloader_cfg is None:
                    self._eval_func = None
                else:
                    self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)

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
            None,
            None,
            self._eval_dataloader,
            self._eval_func,
            _resume)

        try:
            with time_limit(self.conf.usr_cfg.tuning.exit_policy.timeout):
                self.strategy.traverse()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.info("Unexpected exception {} happened during turing.".format(repr(e)))
        finally: 
            if self.strategy.best_qmodel:
                logger.info(
                    "Specified timeout or max trials is reached! "
                    "Found a converted model which meet accuracy goal. Exit.")
                self.strategy.deploy_config()
            else:
                logger.info(
                    "Specified timeout or max trials is reached! "
                    "Not found any converted model which meet accuracy goal. Exit.")

            logger.info("Graph optimization is done. Please invoke model.save() to save " \
                        "optimized model to disk.")

            return self.strategy.best_qmodel

    def dataset(self, dataset_type, *args, **kwargs):
        from .data import DATASETS
        return DATASETS(self.framework)[dataset_type](*args, **kwargs)

    def set_config_by_model(self, model_obj):
        if model_obj.framework() != 'tensorflow':
            logger.warning("Only TensorFlow graph optimization is supported at current stage.")
            sys.exit(0)
        self.conf.usr_cfg.model.framework = model_obj.framework()

        if self._precisions == ['bf16'] and not CpuInfo().bf16:
            if os.getenv('FORCE_BF16') == '1':
                logger.warning("Graph optimization will generate bf16 graph although " \
                               "the hardware doesn't support bf16 instruction.")
            else:
                logger.warning("Graph optimization exits due to the hardware " \
                               "doesn't support bf16 instruction.")
                sys.exit(0)

        self.conf.usr_cfg.graph_optimization.precisions = self._precisions
        self.conf.usr_cfg.model.inputs = self._input
        if isinstance(self._output, str) and ',' in self._output:
            self.conf.usr_cfg.model.outputs = [s.strip() for s in self._output.split(',')]
        else:
            self.conf.usr_cfg.model.outputs = self._output

    @property
    def precisions(self):
        return self._precisions

    @precisions.setter
    def precisions(self, customized_precisions):
        if isinstance(customized_precisions, list):
            self._precisions = sorted([i.strip() for i in customized_precisions])
        elif isinstance(customized_precisions, str):
            self._precisions = sorted([i.strip() for i in customized_precisions.split(',')])


    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, customized_input):
        self._input = customized_input

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, customized_output):
        self._output = customized_output

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
                                      Another good practice is to use neural_compressor.common.DataLoader
                                      to initialize a neural_compressor dataloader object.
                                      Notice neural_compressor.common.DataLoader is just a wrapper of the
                                      information needed to build a dataloader, it can't yield
                                      batched data and only in this setter method
                                      a 'real' eval_dataloader will be created,
                                      the reason is we have to know the framework info
                                      and only after the Quantization object created then
                                      framework infomation can be known. Future we will support
                                      creating iterable dataloader from neural_compressor.common.DataLoader

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
                       (eg, tensorflow frozen_pb or path to a saved model), but not recommended.
                       Best practice is to set from a initialized neural_compressor.common.Model.
                       If tensorflow model is used, model's inputs/outputs will be auto inferred,
                       but sometimes auto inferred inputs/outputs will not meet your requests,
                       set them manually in config yaml file. Another corner case is slim model
                       of tensorflow, be careful of the name of model configured in yaml file,
                       make sure the name is in supported slim model list.

        """

        if not isinstance(user_model, BaseModel):
            logger.warning("Force convert framework model to neural_compressor model.")
            self._model = NCModel(user_model)
        else:
            self._model = user_model

        if self.conf.usr_cfg.model.framework == 'NA':
            self.set_config_by_model(self._model)
            self.framework = self.conf.usr_cfg.model.framework.lower()
            set_backend(self.framework)

    @property
    def metric(self):
        assert False, 'Should not try to get the value of `metric` attribute.'
        return None

    @metric.setter
    def metric(self, user_metric):
        """Set metric class and neural_compressor will initialize this class when evaluation
           neural_compressor have many built-in metrics, but user can set specific metric through
           this api. The metric class should take the outputs of the model or
           postprocess(if have) as inputs, neural_compressor built-in metric always take
           (predictions, labels) as inputs for update,
           and user_metric.metric_cls should be sub_class of neural_compressor.metric.BaseMetric.

        Args:
            user_metric(neural_compressor.common.Metric): user_metric should be object initialized from
                                             neural_compressor.common.Metric, in this method the
                                             user_metric.metric_cls will be registered to
                                             specific frameworks and initialized.

        """
        from .common import Metric as NCMetric
        assert isinstance(user_metric, NCMetric), \
            'please initialize a neural_compressor.common.Metric and set....'

        metric_cfg = {user_metric.name : {**user_metric.kwargs}}
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.metric"):
            logger.warning("Override the value of `metric` field defined in yaml file" \
                           " as user defines the value of `metric` attribute by code.")
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.metric", metric_cfg)
        self.conf.usr_cfg = DotDict(self.conf.usr_cfg)
        from .metric import METRICS
        metrics = METRICS(self.framework)
        metrics.register(user_metric.name, user_metric.metric_cls)

    @property
    def postprocess(self, user_postprocess):
        assert False, 'Should not try to get the value of `postprocess` attribute.'
        return None

    @postprocess.setter
    def postprocess(self, user_postprocess):
        """Set postprocess class and neural_compressor will initialize this class when evaluation.
           The postprocess class should take the outputs of the model as inputs, and
           output (predictions, labels) as inputs for metric update.
           user_postprocess.postprocess_cls should be sub_class of neural_compressor.data.BaseTransform.

        Args:
            user_postprocess(neural_compressor.common.Postprocess):
                user_postprocess should be object initialized from neural_compressor.common.Postprocess,
                in this method the user_postprocess.postprocess_cls will be
                registered to specific frameworks and initialized.

        """
        from .common import Postprocess as NCPostprocess
        assert isinstance(user_postprocess, NCPostprocess), \
            'please initialize a neural_compressor.common.Postprocess and set....'
        postprocess_cfg = {user_postprocess.name : {**user_postprocess.kwargs}}
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.postprocess"):
            logger.warning("Override the value of `postprocess` field defined in yaml file" \
                           " as user defines the value of `postprocess` attribute by code.")
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.postprocess.transform", postprocess_cfg)
        from .data import TRANSFORMS
        postprocesses = TRANSFORMS(self.framework, 'postprocess')
        postprocesses.register(user_postprocess.name, user_postprocess.postprocess_cls)

    @property
    def eval_func(self):
        assert False, 'Should not try to get the value of `eval_func` attribute.'
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
        return 'GraphOptimization'
