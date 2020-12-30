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
import pickle
from .conf.config import Conf
from .conf.dotdict import deep_set, DotDict
from .strategy import STRATEGIES
from .metric import METRICS
from .utils import logger
from .utils.create_obj_from_config import create_dataloader
from .data import DataLoader as DATALOADER
from .data import DATASETS, TRANSFORMS

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
        self.framework = self.conf.usr_cfg.model.framework.lower()

    def __call__(self, model, q_dataloader=None, q_func=None, eval_dataloader=None,
                 eval_func=None):
        """The main entry point of automatic quantization tuning.

           This interface works on all the DL frameworks that lpot supports
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

        Args:
            model (object):                        For Tensorflow model, it could be a path
                                                   to frozen pb,loaded graph_def object or
                                                   a path to ckpt/savedmodel folder.
                                                   For PyTorch model, it's torch.nn.model
                                                   instance.
                                                   For MXNet model, it's mxnet.symbol.Symbol
                                                   or gluon.HybirdBlock instance.
            q_dataloader (generator):              Data loader for calibration, mandatory for
                                                   post-training quantization. It is iterable
                                                   and should yield a tuple (input, label) for
                                                   calibration dataset containing label,
                                                   or yield (input, _) for label-free calibration
                                                   dataset. The input could be a object, list,
                                                   tuple or dict, depending on user implementation,
                                                   as well as it can be taken as model input.
            q_func (function, optional):           Training function for Quantization-Aware
                                                   Training. It is optional and only takes effect
                                                   when user choose "quant_aware_training"
                                                   approach in yaml.
                                                   This function takes "model" as input parameter
                                                   and executes entire training process with self
                                                   contained training hyper-parameters. If this
                                                   parameter specified, eval_dataloader parameter
                                                   plus metric defined in yaml, or eval_func
                                                   parameter should also be specified at same time.
            eval_dataloader (generator, optional): Data loader for evaluation. It is iterable
                                                   and should yield a tuple of (input, label).
                                                   The input could be a object, list, tuple or
                                                   dict, depending on user implementation,
                                                   as well as it can be taken as model input.
                                                   The label should be able to take as input of
                                                   supported metrics. If this parameter is
                                                   not None, user needs to specify pre-defined
                                                   evaluation metrics through configuration file
                                                   and should set "eval_func" paramter as None.
                                                   Tuner will combine model, eval_dataloader
                                                   and pre-defined metrics to run evaluation
                                                   process.
            eval_func (function, optional):        The evaluation function provided by user.
                                                   This function takes model as parameter,
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

        Returns:
            quantized model: best qanitized model found, otherwise return None

        """
        cfg = self.conf.usr_cfg

        # when eval_func is set, will be directly used and eval_dataloader can be None
        if eval_func is None:
            if eval_dataloader is None:
                eval_dataloader_cfg = cfg.evaluation.accuracy.dataloader if \
                                      cfg.evaluation and cfg.evaluation.accuracy else None

                if eval_dataloader_cfg is None:
                    self.eval_func = self._fake_eval_func
                    self.eval_dataloader = None
                else:
                    self.eval_dataloader = create_dataloader(self.framework, \
                                                             eval_dataloader_cfg)
                    self.eval_func = None
            else:
                assert hasattr(eval_dataloader, 'batch_size'), \
                       "eval_dataloader must have batch_size attribute!"
                assert hasattr(eval_dataloader, '__iter__') or \
                       hasattr(eval_dataloader, '__getitem__'), \
                       "eval_dataloader must implement __iter__ or __getitem__ magic method!"
                self.eval_dataloader = eval_dataloader
                self.eval_func = None
        else:
            self.eval_dataloader =None
            self.eval_func = eval_func

        if q_func is None:
            if q_dataloader is None:
                calib_dataloader_cfg = cfg.quantization.calibration.dataloader
                assert calib_dataloader_cfg is not None, \
                       "dataloader field of calibration field of quantization section " \
                       "in yaml file should be configured as q_dataloader is None!"
                self.calib_dataloader = create_dataloader(self.framework, calib_dataloader_cfg)
                self.q_func = None
            else:
                assert hasattr(q_dataloader, 'batch_size'), \
                       "q_dataloader must have batch_size attribute!"
                assert hasattr(q_dataloader, '__iter__') or \
                       hasattr(q_dataloader, '__getitem__'), \
                       "q_dataloader must implement __iter__ or __getitem__ magic method!"
                self.calib_dataloader = q_dataloader
                self.q_func = None
        else:
            self.calib_dataloader =None
            self.q_func = q_func

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
            model,
            self.conf,
            self.calib_dataloader,
            self.q_func,
            self.eval_dataloader,
            self.eval_func,
            _resume)

        self.strategy.traverse()

        if self.strategy.best_qmodel:
            logger.info(
                "Specified timeout or max trials is reached! "
                "Found a quantized model which meet accuracy goal. Exit...")
            self.strategy.deploy_config()
        else:
            logger.info(
                "Specified timeout or max trials is reached! "
                "Not found any quantized model which meet accuracy goal. Exit...")

        return self.strategy.best_qmodel

    def dataset(self, dataset_type, *args, **kwargs):
        return DATASETS(self.framework)[dataset_type](*args, **kwargs)

    def dataloader(self, dataset, batch_size=1, collate_fn=None, last_batch='rollover',
                   sampler=None, batch_sampler=None, num_workers=0, pin_memory=False):
        return DATALOADER(framework=self.framework, dataset=dataset,
                          batch_size=batch_size, collate_fn=collate_fn, last_batch=last_batch,
                          sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
                          pin_memory=pin_memory)

    def metric(self, name, metric_cls, **kwargs):
        metric_cfg = {name : {**kwargs}} 
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.metric", metric_cfg)
        self.conf.usr_cfg = DotDict(self.conf.usr_cfg)
        metrics = METRICS(self.framework)
        metrics.register(name, metric_cls)
        
    def postprocess(self, name, postprocess_cls, **kwargs):
        postprocess_cfg = {name : {**kwargs}} 
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.postprocess.transform", postprocess_cfg)
        postprocesses = TRANSFORMS(self.framework, 'postprocess')
        postprocesses.register(name, postprocess_cls)
        logger.info("{} registered to postprocess".format(name))

    # if user doesn't config evaluation dataloader in yaml and eval_func is None, a
    # fake eval func is created to do quantization once without tuning
    def _fake_eval_func(self, model):
        return 1.

