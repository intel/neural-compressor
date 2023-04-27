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
import os
import sys
import pickle
import numpy as np
import random

from neural_compressor.data.dataloaders.dataloader import check_dataloader
from neural_compressor.metric.metric import register_customer_metric
from .utils.utility import time_limit
from .strategy import STRATEGIES
from .config import _Config, options
from .utils import logger
from .model.model import wrap_model_from


def fit(model,
        config=None,
        eval_func=None,
        eval_dataloader=None,
        eval_metric=None,
        **kwargs):
    """Fit low precision model generation across multiple framework backends.

    Args:
        model (object):                       For Tensorflow model, it could be a path
                                              to frozen pb, loaded graph_def object or
                                              a path to ckpt/savedmodel folder.
                                              For PyTorch model, it's torch.nn.model
                                              instance. For onnx model, it chould be a path
                                              to .onnx file or onnx.onnx_ml_pb2.ModelProto.
                                              For MXNet model, it's mxnet.symbol.Symbol
                                              or gluon.HybirdBlock instance.
        config (MixedPrecisionConfig):        The MixedPrecisionConfig class containing accuracy goal,
                                              tuning objective and mixed_precision tuning space etc.
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
                                              mixed_precision.

    Returns:
        A Mixed precision model across various DL frameworks.

    Raises:
        AssertionError.

    Example::

        from neural_compressor import mix_precision
        from neural_compressor.config import MixedPrecisionConfig

        conf = MixedPrecisionConfig()
        converted_model = mix_precision.fit(model, config=conf)
    """
    if eval_dataloader is not None:
        check_dataloader(eval_dataloader)

    if config.precisions in config.excluded_precisions:
        logger.warning("Target precision is in excluded_precisions, "
                       "please modify precision or excluded_precisions to make it understandable.")
        sys.exit(0)

    wrapped_model = wrap_model_from(model, config)

    if eval_metric is not None:
        metric = register_customer_metric(eval_metric, config.framework)
    else:
        metric = None

    conf = _Config(mixed_precision=config,
                   quantization=None,
                   benchmark=None,
                   pruning=None,
                   distillation=None,
                   nas=None)
    seed = options.random_seed
    random.seed(seed)
    np.random.seed(seed)

    _resume = None
    # check if interrupted tuning procedure exists. if yes, it will resume the
    # whole auto tune process.
    resume_file = os.path.abspath(os.path.expanduser(
        options.resume_from)) if options.workspace and options.resume_from else None
    if resume_file:
        assert os.path.exists(resume_file), \
            "The specified resume file {} doesn't exist!".format(resume_file)
        with open(resume_file, 'rb') as f:
            _resume = pickle.load(f).__dict__

    strategy = STRATEGIES['automixedprecision'](
        model=wrapped_model,
        conf=conf,
        eval_func=eval_func,
        eval_dataloader=eval_dataloader,
        eval_metric=metric,
        resume=_resume,
        q_hooks=None)

    try:
        with time_limit(config.tuning_criterion.timeout):
            strategy.traverse()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Unexpected exception {} happened during tuning.".format(repr(e)))
        import traceback
        traceback.print_exc()
    finally:
        if strategy.best_qmodel:
            logger.info(
                "Specified timeout or max trials is reached! "
                "Found a quantized model which meet accuracy goal. Exit.")
            strategy.deploy_config()
        else:
            logger.error(
                "Specified timeout or max trials is reached! "
                "Not found any quantized model which meet accuracy goal. Exit.")

        return strategy.best_qmodel
