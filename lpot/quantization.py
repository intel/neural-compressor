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
from .utils import logger
from .data import DATALOADERS, DATASETS
from .experimental import Quantization as ExpQuantization

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
        self.exp_quantizer = ExpQuantization(conf_fname)

    def __call__(self, model, q_dataloader=None, q_func=None, eval_dataloader=None,
                 eval_func=None):
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

              After that, User specifies fp32 "model", calibration dataset "q_dataloader"
              and evaluation dataset "eval_dataloader".
              The calibrated and quantized model is evaluated with "eval_dataloader"
              with evaluation metrics specified in the configuration file. The evaluation tells
              the tuner whether the quantized model meets the accuracy criteria. If not,
              the tuner starts a new calibration and tuning flow.

              For this usage, model, q_dataloader and eval_dataloader parameters are mandatory.

           c) Partial yaml configuration: User specifies dataloaders used in calibration phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The calibrated and quantized model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the quantized model meets
              the accuracy criteria. If not, the Tuner starts a new calibration and tuning flow.

              For this usage, model, q_dataloader and eval_func parameters are mandatory.

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

        logger.warning('This API is going to be deprecated, please import '
            'lpot.experimental.Quantization, set the attributes about '
            'dataloader and metric, then use new __call__ method')
        
        self.exp_quantizer.model = model
        if q_dataloader is not None:
            self.exp_quantizer.calib_dataloader = q_dataloader 
        elif q_func is not None:
            self.exp_quantizer.q_func = q_func

        if eval_func is not None:
            self.exp_quantizer.eval_func = eval_func 
        elif eval_dataloader is not None:
            self.exp_quantizer.eval_dataloader = eval_dataloader

        lpot_model = self.exp_quantizer()
        if self.exp_quantizer.framework == 'tensorflow':
            return lpot_model.graph if lpot_model else None
        if self.exp_quantizer.framework == 'pytorch':
            saved_path = os.path.abspath(os.path.join(os.path.expanduser(
                self.exp_quantizer.conf.usr_cfg.tuning.workspace.path), 'checkpoint'))
            lpot_model.save(saved_path)
        return lpot_model.model

    def dataset(self, dataset_type, *args, **kwargs):
        return DATASETS(self.exp_quantizer.framework)[dataset_type](*args, **kwargs)

    def dataloader(self, dataset, batch_size=1, collate_fn=None, last_batch='rollover',
                   sampler=None, batch_sampler=None, num_workers=0, pin_memory=False):
        return DATALOADERS[self.exp_quantizer.framework](dataset=dataset,
                          batch_size=batch_size, collate_fn=collate_fn, last_batch=last_batch,
                          sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
                          pin_memory=pin_memory)

    def metric(self, name, metric_cls, **kwargs):
        from .experimental.common import Metric as LpotMetric
        lpot_metric = LpotMetric(metric_cls, name, **kwargs)
        self.exp_quantizer.metric = lpot_metric

    def postprocess(self, name, postprocess_cls, **kwargs):
        from .experimental.common import Postprocess as LpotPostprocess
        lpot_postprocess = LpotPostprocess(postprocess_cls, name, **kwargs)
        self.exp_quantizer.postprocess = lpot_postprocess

