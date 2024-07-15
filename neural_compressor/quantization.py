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
import os
import pickle
import random

import numpy as np

from .config import _Config, options
from .data import check_dataloader
from .metric import register_customer_metric
from .model import Model
from .strategy import STRATEGIES
from .utils import logger
from .utils.utility import dump_class_attrs, time_limit


def fit(
    model,
    conf,
    calib_dataloader=None,
    calib_func=None,
    eval_func=None,
    eval_dataloader=None,
    eval_metric=None,
    **kwargs,
):
    """Quantize the model with a given configure.

    Args:
        model (torch.nn.Module):              For Tensorflow model, it could be a path
                                              to frozen pb,loaded graph_def object or
                                              a path to ckpt/savedmodel folder.
                                              For PyTorch model, it's torch.nn.model
                                              instance.
                                              For MXNet model, it's mxnet.symbol.Symbol
                                              or gluon.HybirdBlock instance.
        conf (PostTrainingQuantConfig):       The class of PostTrainingQuantConfig containing accuracy goal,
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
                                              and executes entire inference process.
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
                                                   return accuracy.
                                              The user only needs to set eval_func or
                                              eval_dataloader and eval_metric which is an alternative option
                                              to tune the model accuracy.
        eval_dataloader (generator, optional): Data loader for evaluation. It is iterable
                                              and should yield a tuple of (input, label).
                                              The input could be a object, list, tuple or
                                              dict, depending on user implementation,
                                              as well as it can be taken as model input.
                                              The label should be able to take as input of
                                              supported metrics. If this parameter is
                                              not None, user needs to specify pre-defined
                                              evaluation metrics through configuration file
                                              and should set "eval_func" parameter as None.
                                              Tuner will combine model, eval_dataloader
                                              and pre-defined metrics to run evaluation
                                              process.
        eval_metric (dict or obj):            Set metric class or a dict of built-in metric configures,
                                              and neural_compressor will initialize this class when evaluation.

            1. neural_compressor have many built-in metrics,
               user can pass a metric configure dict to tell neural compressor what metric will be use.
               You also can set multi-metrics to evaluate the performance of a specific model.
                    Single metric:
                        {topk: 1}
                    Multi-metrics:
                        {topk: 1,
                         MSE: {compare_label: False},
                         weight: [0.5, 0.5],
                         higher_is_better: [True, False]
                        }
        For the built-in metrics, please refer to below link:
        https://github.com/intel/neural-compressor/blob/master/docs/source/metric.md#supported-built-in-metric-matrix.

            2. User also can get the built-in metrics by neural_compressor.Metric:
                Metric(name="topk", k=1)
            3. User also can set specific metric through this api. The metric class should take the outputs of
               the model or postprocess(if have) as inputs, neural_compressor built-in metric always
               take (predictions, labels) as inputs for update, and user_metric.metric_cls should be
               sub_class of neural_compressor.metric.BaseMetric.

    Example::

        # Quantization code for PTQ
        from neural_compressor import PostTrainingQuantConfig
        from neural_compressor import quantization
        def eval_func(model):
            for input, label in dataloader:
                output = model(input)
                metric.update(output, label)
            accuracy = metric.result()
            return accuracy

        conf = PostTrainingQuantConfig()
        q_model = quantization.fit(model_origin,
                                   conf,
                                   calib_dataloader=dataloader,
                                   eval_func=eval_func)

        # Saved quantized model in ./saved folder
        q_model.save("./saved")
    """
    _raw_model = model

    if calib_dataloader is not None:
        check_dataloader(calib_dataloader)
    if eval_dataloader is not None:
        check_dataloader(eval_dataloader)

    seed = options.random_seed
    random.seed(seed)
    np.random.seed(seed)
    wrapped_model = Model(model, conf=conf)

    if eval_metric is not None:
        metric = register_customer_metric(eval_metric, conf.framework)
    else:
        metric = None

    config = _Config(quantization=conf, benchmark=None, pruning=None, distillation=None, nas=None)
    strategy_name = conf.tuning_criterion.strategy

    if conf.quant_level == "auto":
        strategy_name = "auto"
    elif conf.quant_level == 0:
        strategy_name = "conservative"

    if strategy_name == "mse_v2":
        if not (
            conf.framework.startswith("tensorflow") or conf.framework in ["pytorch_fx", "onnxruntime"]
        ):  # pragma: no cover
            strategy_name = "basic"
            logger.warning(f"MSE_v2 does not support {conf.framework} now, use basic instead.")
            logger.warning("Only tensorflow, pytorch_fx is supported by MSE_v2 currently.")
    assert strategy_name in STRATEGIES, "Tuning strategy {} is NOT supported".format(strategy_name)

    logger.info(f"Start {strategy_name} tuning.")
    _resume = None
    # check if interrupted tuning procedure exists. if yes, it will resume the
    # whole auto tune process.
    resume_file = (
        os.path.abspath(os.path.expanduser(options.resume_from)) if options.workspace and options.resume_from else None
    )
    if resume_file:
        assert os.path.exists(resume_file), "The specified resume file {} doesn't exist!".format(resume_file)
        with open(resume_file, "rb") as f:
            _resume = pickle.load(f).__dict__

    if eval_func is None and eval_dataloader is None:  # pragma: no cover
        logger.info("Quantize model without tuning!")

    strategy = STRATEGIES[strategy_name](
        model=wrapped_model,
        conf=config,
        q_dataloader=calib_dataloader,
        q_func=calib_func,
        eval_func=eval_func,
        eval_dataloader=eval_dataloader,
        eval_metric=metric,
        resume=_resume,
        q_hooks=None,
    )

    try:
        with time_limit(conf.tuning_criterion.timeout):
            logger.debug("Dump user configuration:")
            conf_dict = {}
            dump_class_attrs(conf, conf_dict)
            import copy

            def update(d):
                o = copy.copy(d)
                for k, v in o.items():
                    if k == "example_inputs":
                        o[k] = "Not printed here due to large size tensors..."
                    elif isinstance(v, dict):
                        o[k] = update(v)
                return o

            logger.info(update(conf_dict))

            strategy.traverse()

    except KeyboardInterrupt:
        pass
    except Exception as e:  # pragma: no cover
        logger.error("Unexpected exception {} happened during tuning.".format(repr(e)))
        import traceback

        traceback.print_exc()
    finally:
        if strategy.get_best_qmodel():
            logger.info(
                "Specified timeout or max trials is reached! " "Found a quantized model which meet accuracy goal. Exit."
            )
            strategy.deploy_config()
        else:
            logger.error(
                "Specified timeout or max trials is reached! "
                "Not found any quantized model which meet accuracy goal. Exit."
            )

        return strategy.get_best_qmodel()
