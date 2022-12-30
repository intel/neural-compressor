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

"""Neural Compressor Quantization API."""

from .experimental import Quantization as ExpQuantization
from neural_compressor.conf.pythonic_config import Config
from neural_compressor.config import PostTrainingQuantConfig

def fit(model,
        conf,
        calib_dataloader=None,
        calib_func=None,
        eval_dataloader=None,
        eval_func=None,
        eval_metric=None,
        **kwargs):
    """Quantize the model with a given configure.

    Args:
        model (torch.nn.Module):              For Tensorflow model, it could be a path
                                              to frozen pb,loaded graph_def object or
                                              a path to ckpt/savedmodel folder.
                                              For PyTorch model, it's torch.nn.model
                                              instance.
                                              For MXNet model, it's mxnet.symbol.Symbol
                                              or gluon.HybirdBlock instance.
        conf (string or obj):                 The path to the YAML configuration file or
                                              QuantConf class containing accuracy goal,
                                              tuning objective and preferred calibration &
                                              quantization tuning space etc.
        calib_dataloader (generator):         Data loader for calibration, mandatory for
                                              post-training quantization. It is iterable
                                              and should yield a tuple (input, label) for
                                              calibration dataset containing label,
                                              or yield (input, _) for label-free calibration
                                              dataset. The input could be a object, list,
                                              tuple or dict, depending on user implementation,
                                              as well as it can be taken as model input.
        calib_func (function, optional):      Calibration function for post-training static
                                              quantization. It is optional.
                                              This function takes "model" as input parameter
                                              and executes entire inference process. If this
                                              parameter specified, calib_dataloader is also needed
                                              for FX trace if PyTorch >= 1.13.
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
        eval_func (function, optional):       The evaluation function provided by user.
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
        eval_metric (str or obj):             Set metric class and neural_compressor will initialize 
                                              this class when evaluation.
    """
    if isinstance(conf, PostTrainingQuantConfig):
        if eval_func is None and eval_dataloader is None:
            conf.performance_only = True
        conf = Config(quantization=conf, benchmark=None, pruning=None, distillation=None, nas=None)
    quantizer = ExpQuantization(conf)
    quantizer.model = model
    if eval_func is not None:
        quantizer.eval_func = eval_func
    if calib_dataloader is not None:
        quantizer.calib_dataloader = calib_dataloader
    if calib_func is not None:
        quantizer.calib_func = calib_func
    if eval_dataloader is not None:
        quantizer.eval_dataloader = eval_dataloader
    if eval_metric is not None:
        quantizer.metric = eval_metric
    return quantizer()
