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
from .conf.config import QuantConf
from .conf.dotdict import deep_get, deep_set, DotDict
from .conf.pythonic_config import Config
from .model.model import BaseModel, get_model_fwk_name, get_model_type, Model, MODELS
from .strategy import STRATEGIES
from .utils import logger
from .utils.utility import time_limit


class _PostTrainingQuant:
    """Post Training Quantization class.

    It automatically searches for optimal quantization recipes for low precision model inference,
    achieving best tuning objectives like inference performance within accuracy loss constraints.
    Tuner abstracts out the differences of quantization APIs across various DL frameworks
    and brings a unified API for automatic quantization that works on frameworks including
    tensorflow, pytorch and mxnet.
    Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria
    (<1% or <0.1% etc.) and tuning objectives (performance, memory footprint etc.).

    Example::

        conf = PostTrainingQuantConfig()
        quantizer = _PostTrainingQuant(conf)
        quantizer.model = model
        quantizer.eval_func = eval_func
        quantizer.calib_dataloader = calib_dataloader
        q_model = quantizer.fit()
    """
    def __init__(self, conf, **kwargs):
        """Initialize the parameters.

        Args:
            conf (PostTrainingQuantConfig): A instance of PostTrainingQuantConfig to
                                            specify the quantization behavior.
        """
        conf = Config(quantization=conf, benchmark=None, pruning=None, distillation=None, nas=None)
        self.conf = QuantConf()
        self.conf.map_pyconfig_to_cfg(conf)
        seed = self.conf.usr_cfg.tuning.random_seed
        random.seed(seed)
        np.random.seed(seed)
        self._train_func = None
        self._calib_dataloader = None
        self._eval_func = None
        self._eval_dataloader = None
        self._eval_metric = None
        self._model = None
        self.callbacks = None
        if "model" in kwargs:
            self.model = kwargs["model"]

    def pre_proccess(self):
        """Create strategy to optimize model."""
        cfg = self.conf.usr_cfg

        if os.environ.get("PERFORMANCE_ONLY") in ['0', '1']:
            performance_only = bool(int(os.environ.get("PERFORMANCE_ONLY")))
            deep_set(cfg, 'tuning.exit_policy.performance_only', performance_only)
            logger.info("Get environ 'PERFORMANCE_ONLY={}'," \
                " force setting 'tuning.exit_policy.performance_only = True'.".format(performance_only))

        strategy = cfg.tuning.strategy.name.lower()
        
        if cfg.quantization.quant_level == "auto":
            strategy = "auto"
            
        elif cfg.quantization.quant_level == 0:
            strategy = "conservative"

        if strategy == "mse_v2":
            if not (cfg.model.framework.startswith("tensorflow") or cfg.model.framework == 'pytorch_fx'):
                strategy = "basic"
                logger.warning(f"MSE_v2 does not support {cfg.model.framework} now, use basic instead.")
                logger.warning("Only tensorflow, pytorch_fx is supported by MSE_v2 currently.")
        assert strategy in STRATEGIES, "Tuning strategy {} is NOT supported".format(strategy)

        logger.info(f"Start {strategy} tuning.")
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

        if self._eval_func is None and self._eval_dataloader is None:
            self.conf.usr_cfg.tuning.exit_policy.performance_only = True
            logger.info("Quantize model without tuning!")

        self.strategy = STRATEGIES[strategy](
            self._model,
            self.conf,
            self._calib_dataloader,
            self._train_func,
            self._eval_dataloader,
            self._eval_func,
            _resume,
            self.callbacks.hooks if self.callbacks is not None else None)

    def execute(self):
        """Quantization execute routinue based on strategy design."""
        try:
            with time_limit(self.conf.usr_cfg.tuning.exit_policy.timeout):
                logger.debug("Dump user configuration:")
                logger.debug(self.conf.usr_cfg)
                self.strategy.traverse()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error("Unexpected exception {} happened during tuning.".format(repr(e)))
            import traceback
            traceback.print_exc()
        finally:
            if self.strategy.best_qmodel:
                logger.info(
                    "Specified timeout or max trials is reached! "
                    "Found a quantized model which meet accuracy goal. Exit.")
                self.strategy.deploy_config()
            else:
                logger.error(
                    "Specified timeout or max trials is reached! "
                    "Not found any quantized model which meet accuracy goal. Exit.")

            return self.strategy.best_qmodel

    def __call__(self):
        """Execute this class.

        For derived classes(Pruning, Quantization, etc.), an override function is required.
        """
        self.pre_proccess()
        results = self.execute()
        return results

    fit = __call__

    @property
    def model(self):
        """Getter of model in neural_compressor.model."""
        return self._model

    @model.setter
    def model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object.

        Args:
            user_model: user are supported to set model from original framework model format
                        (eg, tensorflow frozen_pb or path to a saved model),
                        but not recommended. Best practice is to set from a initialized
                        neural_compressor.model.Model.
                        If tensorflow model is used, model's inputs/outputs will be
                        auto inferenced, but sometimes auto inferenced
                        inputs/outputs will not meet your requests,
                        set them manually in config yaml file.
                        Another corner case is slim model of tensorflow,
                        be careful of the name of model configured in yaml file,
                        make sure the name is in supported slim model list.

        """
        cfg = self.conf.usr_cfg
        if cfg.model.framework == 'NA':
            if isinstance(user_model, BaseModel):
                cfg.model.framework = list(MODELS.keys())[list(MODELS.values()).index(type(user_model))]
                if cfg.model.backend == "ipex":
                    assert cfg.model.framework == "pytorch_ipex", "Please wrap the model with correct Model class!"
                if cfg.model.backend == "itex":
                    if get_model_type(user_model.model) == 'keras':
                        assert cfg.model.framework == "keras", "Please wrap the model with KerasModel class!"
                    else:
                        assert cfg.model.framework == "pytorch_itex", \
                            "Please wrap the model with TensorflowModel class!"
            else:
                framework = get_model_fwk_name(user_model)
                cfg.model.framework = framework
                if framework == "tensorflow":
                    if get_model_type(user_model) == 'keras' and cfg.model.backend == 'itex':
                        cfg.model.framework = 'keras'
                if framework == "pytorch":
                    if cfg.model.backend == "default":
                        cfg.model.framework = "pytorch_fx"
                    elif cfg.model.backend == "ipex":
                        cfg.model.framework = "pytorch_ipex"

        if not isinstance(user_model, BaseModel):
            logger.warning("Force convert framework model to neural_compressor model.")
            if "tensorflow" in cfg.model.framework or cfg.model.framework == "keras":
                self._model = Model(user_model, backend=cfg.model.framework, device=cfg.device)
            else:
                self._model = Model(user_model, backend=cfg.model.framework)
        else:
            if cfg.model.framework == "pytorch_ipex":
                from neural_compressor.model.torch_model import IPEXModel
                assert type(user_model) == IPEXModel, \
                            "The backend is ipex, please wrap the model with IPEXModel class!"
            elif cfg.model.framework == "pytorch_fx":
                from neural_compressor.model.torch_model import PyTorchFXModel
                assert type(user_model) == PyTorchFXModel, \
                            "The backend is default, please wrap the model with PyTorchFXModel class!"

            self._model = user_model

        if 'tensorflow' in cfg.model.framework:
            self._model.name = cfg.model.name
            self._model.output_tensor_names = cfg.model.outputs
            self._model.input_tensor_names = cfg.model.inputs
            self._model.workspace_path = cfg.tuning.workspace.path

    @property
    def eval_func(self):
        """Not support get eval_func."""
        assert False, 'Should not try to get the value of `eval_func` attribute.'

    @eval_func.setter
    def eval_func(self, user_eval_func):
        """Eval function for component.

        Args:
            user_eval_func: This function takes "model" as input parameter
                         and executes entire evaluation process with self
                         contained metrics. If eval_func set,
                         an evaluation process must be triggered
                         to make evaluation of the model executed.
        """
        self._eval_func = user_eval_func

    @property
    def eval_dataloader(self):
        """Getter to eval dataloader."""
        return self._eval_dataloader

    @eval_dataloader.setter
    def eval_dataloader(self, dataloader):
        """Set Data loader for evaluation of component.

        It is iterable and the batched data should consists of yield (input, _).
        the input in the batched data will be used for model inference, so it
        should satisfy the input format of specific model.

        Args:
            dataloader(generator): user are supported to set a user defined dataloader
                                   which meet the requirements that can yield tuple of
                                   (input, label)/(input, _) batched data.
        """
        assert hasattr(dataloader, '__iter__') and \
            hasattr(dataloader, 'batch_size'), \
            'dataloader must implement __iter__ method and batch_size attribute'

        self._eval_dataloader = dataloader

    @property
    def metric(self):
        """Get `metric` attribute."""
        assert False, 'Should not try to get the value of `metric` attribute.'
        return None

    @metric.setter
    def metric(self, user_metric):
        """Set metric class or a dict of built-in metric configures.

        1. neural_compressor have many built-in metrics,
           user can pass a metric configure dict to tell neural compressor what metric will be use.
           You can set multi-metrics to evaluate the performance of a specific model.
                Single metric:
                    {topk: 1}
                Multi-metrics:
                    {topk: 1,
                     MSE: {compare_label: False},
                    }
        For the built-in metrics, please refer to below link:
        https://github.com/intel/neural-compressor/blob/master/docs/source/metric.md#supported-built-in-metric-matrix.

        2. User also can set specific metric through this api. The metric class should take the outputs of the model or
           postprocess(if have) as inputs, neural_compressor built-in metric always take(predictions, labels)
           as inputs for update, and user_metric.metric_cls should be sub_class of neural_compressor.metric.BaseMetric.

        Args:
            user_metric(neural_compressor.metric.Metric or a dict of built-in metric configurations):
                The object of Metric or a dict of built-in metric configurations.

        """
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.metric"):
            logger.warning("Override the value of `metric` field defined in yaml file" \
                           " as user defines the value of `metric` attribute by code.")

        from .metric import Metric as NCMetric, METRICS
        if isinstance(user_metric, dict):
            metric_cfg = user_metric
        else:
            if isinstance(user_metric, NCMetric):
                name = user_metric.name
                metric_cls = user_metric.metric_cls
                metric_cfg = {name: {**user_metric.kwargs}}
            else:
                for i in ['reset', 'update', 'result']:
                    assert hasattr(user_metric, i), 'Please realise {} function' \
                                                    'in user defined metric'.format(i)
                metric_cls = type(user_metric).__name__
                name = 'user_' + metric_cls
                metric_cfg = {name: id(user_metric)}
            metrics = METRICS(self.conf.usr_cfg.model.framework)
            metrics.register(name, metric_cls)

        deep_set(self.conf.usr_cfg, "evaluation.accuracy.metric", metric_cfg)
        self.conf.usr_cfg = DotDict(self.conf.usr_cfg)

        self._metric = user_metric

    @property
    def calib_func(self):
        """Not support get train_func."""
        assert False, 'Should not try to get the value of `train_func` attribute.'

    @calib_func.setter
    def calib_func(self, calib_func):
        """Calibrate scale and zero for quantization.

        Args:
            calib_func: This function takes "model" as input parameter
                         and executes entire evaluation process. If calib_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func
                         to make evaluation of the model executed.
        """
        self._train_func = calib_func

    @property
    def calib_dataloader(self):
        """Get `calib_dataloader` attribute."""
        return self._calib_dataloader

    @calib_dataloader.setter
    def calib_dataloader(self, dataloader):
        """Set Data loader for calibration, mandatory for post-training quantization.

        If calib_func is not be set then user must set calibration dataloader,
        and calibration is iterable and the batched data should consists of a tuple like
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
                                    practice is to use neural_compressor.data.DataLoader
                                    to initialize a neural_compressor dataloader object. Notice
                                    neural_compressor.data.DataLoader is just a wrapper of the
                                    information needed to build a dataloader, it can't yield
                                    batched data and only in this setter method
                                    a 'real' calib_dataloader will be created,
                                    the reason is we have to know the framework info
                                    and only after the Quantization object created then
                                    framework infomation can be known.
                                    Future we will support creating iterable dataloader
                                    from neural_compressor.data.DataLoader
        """
        assert hasattr(dataloader, '__iter__') and \
            hasattr(dataloader, 'batch_size'), \
            'dataloader must implement __iter__ method and batch_size attribute'
        self._calib_dataloader = dataloader


def fit(model,
        conf,
        calib_dataloader=None,
        calib_func=None,
        eval_func=None,
        eval_dataloader=None,
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
                                              and executes entire inference process. If this
                                              parameter specified, calib_dataloader is also needed
                                              for FX trace if PyTorch >= 1.13.
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
        eval_metric (dict or obj):             Set metric class or a dict of built-in metric configures,
                                              and neural_compressor will initialize this class when evaluation.

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
    quantizer = _PostTrainingQuant(conf)
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
