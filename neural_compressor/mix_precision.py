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
import pickle
import random
import sys

import numpy as np

from neural_compressor.data import check_dataloader
from neural_compressor.metric import register_customer_metric

from .config import _Config, options
from .model import Model
from .strategy import STRATEGIES
from .utils import alias_param, logger
from .utils.utility import CpuInfo, time_limit


@alias_param("conf", param_alias="config")
def fit(model, conf, eval_func=None, eval_dataloader=None, eval_metric=None, **kwargs):
    """Fit low precision model generation across multiple framework backends.

    Args:
        model (object):                       For Tensorflow model, it could be a path
                                              to frozen pb, loaded graph_def object or
                                              a path to ckpt/savedmodel folder.
                                              For PyTorch model, it's torch.nn.model
                                              instance. For onnx model, it should be a path
                                              to .onnx file or onnx.onnx_ml_pb2.ModelProto.
                                              For MXNet model, it's mxnet.symbol.Symbol
                                              or gluon.HybirdBlock instance.
        conf (MixedPrecisionConfig):        The MixedPrecisionConfig class containing accuracy goal,
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
                                              and should set "eval_func" parameter as None.
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
        converted_model = mix_precision.fit(model, conf=conf)
    """
    if eval_dataloader is not None:
        check_dataloader(eval_dataloader)

    if conf.precisions in conf.excluded_precisions:
        logger.warning(
            "Target precision is in excluded_precisions, "
            "please modify precision or excluded_precisions to make it understandable."
        )
        sys.exit(0)

    wrapped_model = Model(model, conf=conf)

    precisions = list(set(conf.precisions) - set(conf.excluded_precisions))
    if ("bf16" in precisions or "fp16" in precisions) and conf.framework == "onnxruntime":  # pragma: no cover
        if "fp16" in precisions and not (conf.device == "gpu" and conf.backend == "onnxrt_cuda_ep"):
            logger.warning(
                "Mix precision exits due to fp16 for onnx models" "needs 'gpu' device and 'onnxrt_cuda_ep' backend."
            )
            sys.exit(0)
        elif "bf16" in precisions and (
            not (conf.backend == "onnxrt_cuda_ep" and conf.device == "gpu")
            and not (conf.backend == "onnxrt_dnnl_ep" and conf.device == "cpu")
        ):
            logger.warning(
                "Mix precision exits due to bf16 for onnx models needs "
                "'gpu' device and 'onnxrt_cuda_ep' backend, or 'cpu' device and 'onnxrt_dnnl_ep' backend."
            )
            sys.exit(0)
    elif "bf16" in precisions and not CpuInfo().bf16 and conf.framework != "onnxruntime":  # pragma: no cover
        if os.getenv("FORCE_BF16") == "1":
            logger.warning(
                "Mix precision will generate bf16 graph although " "the hardware doesn't support bf16 instruction."
            )
        else:
            logger.warning("Mix precision exits due to the hardware " "doesn't support bf16 instruction.")
            sys.exit(0)
    elif "fp16" in precisions and conf.framework != "onnxruntime":
        logger.warning("Currently mix precision only supports fp16 for onnx models.")
        sys.exit(0)

    if eval_metric is not None:
        metric = register_customer_metric(eval_metric, conf.framework)
    else:
        metric = None

    config = _Config(mixed_precision=conf, quantization=None, benchmark=None, pruning=None, distillation=None, nas=None)
    seed = options.random_seed
    random.seed(seed)
    np.random.seed(seed)

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

    strategy = STRATEGIES["automixedprecision"](
        model=wrapped_model,
        conf=config,
        eval_func=eval_func,
        eval_dataloader=eval_dataloader,
        eval_metric=metric,
        resume=_resume,
        q_hooks=None,
    )

    try:
        with time_limit(conf.tuning_criterion.timeout):
            strategy.traverse()
    except KeyboardInterrupt:
        pass
    except Exception as e:
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
