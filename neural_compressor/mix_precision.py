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
"""Mix Precision for Neural Compressor."""
from .experimental.mixed_precision import MixedPrecision
from neural_compressor.conf.pythonic_config import Config
from neural_compressor.config import MixedPrecisionConfig

def fit(model, config=None, eval_func=None, eval_dataloader=None, eval_metric=None, **kwargs):
    """Fit low precision model generation across multiple framework backends.

    Args:
        model (torch.nn.Module):              For Tensorflow model, it could be a path
                                              to frozen pb, loaded graph_def object or
                                              a path to ckpt/savedmodel folder.
                                              For PyTorch model, it's torch.nn.model
                                              instance.
                                              For MXNet model, it's mxnet.symbol.Symbol
                                              or gluon.HybirdBlock instance.
        config (string or obj):               The path to the YAML configuration file or
                                              QuantConf class containing accuracy goal,
                                              tuning objective and preferred calibration &
                                              quantization tuning space etc.
        eval_func (function, optional):       The evaluation function provided by user.
                                              This function takes model as parameter,
                                              and evaluation dataset and metrics should be
                                              encapsulated in this function implementation
                                              and outputs a higher-is-better accuracy scalar
                                              value.
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
        eval_metric (obj, optional):          An Accuracy object that measures metric for
                                              quantization.

    Returns:
        A MixedPrecision object that generates low precision model across various DL frameworks.

    Raises:
        AssertionError.
    """
    assert isinstance(config, MixedPrecisionConfig), "Please provide MixedPrecisionConfig!"
    conf = Config(quantization=config)
    converter = MixedPrecision(conf)
    precisions = ["bf16", "fp32"]
    precisions = list(set(precisions) - set(config.excluded_precisions))
    converter.precisions = precisions
    converter.model = model
    if eval_func is not None:
        converter.eval_func = eval_func
    if eval_dataloader is not None:
        converter.eval_dataloader = eval_dataloader
    if eval_metric is not None:
        converter.metric = eval_metric
    return converter()
