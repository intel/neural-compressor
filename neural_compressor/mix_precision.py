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
from .utils.utility import time_limit, CpuInfo
from .strategy import STRATEGIES
from .conf.pythonic_config import Config
from .conf.config import MixedPrecision_Conf
from .utils import logger
from .conf.dotdict import deep_get, deep_set, DotDict
from .model.model import BaseModel, get_model_fwk_name, Model, MODELS

class _MixedPrecision:
    """Class used for generating low precision model.

    _MixedPrecision class automatically generates low precision model across various DL
    frameworks including tensorflow, pytorch and onnxruntime.

    Example::

        from neural_compressor.config import MixedPrecisionConfig
        def eval_func(model):
            ...
        return accuracy
        
        conf = MixedPrecisionConfig()
        output_model = mix_precision.fit(
            model,
            conf,
            eval_func=eval_func,
        )
    """
    def __init__(self, conf=None):
        """Initialize `MixedPrecision` class.

        Args:
            conf (obj): The MixedPrecisionConfig class containing accuracy goal, tuning objective etc.
        """
        conf = Config(quantization=conf, benchmark=None, pruning=None, distillation=None, nas=None)
        self.conf = MixedPrecision_Conf()
        self.conf.map_pyconfig_to_cfg(conf)
        seed = self.conf.usr_cfg.tuning.random_seed
        random.seed(seed)
        np.random.seed(seed)

        self._eval_func = None
        self._eval_dataloader = None
        self._eval_metric = None
        self._model = None

    def pre_process(self):
        """Create strategy object for tuning."""
        cfg = self.conf.usr_cfg
        strategy = 'automixedprecision'
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

    def execute(self):
        """Execute routinue based on strategy design."""
        try:
            with time_limit(self.conf.usr_cfg.tuning.exit_policy.timeout):
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

        For derived classes, an override function is required.
        """
        self.pre_process()
        results = self.execute()
        return results

    fit = __call__

    @property
    def precisions(self):
        """Get private member variable `precisions` of `_MixedPrecision` class."""
        return self._precisions

    @precisions.setter
    def precisions(self, customized_precisions):
        """Set private member variable `precisions` of `_MixedPrecision` class."""
        if isinstance(customized_precisions, list):
            self._precisions = sorted([i.strip() for i in customized_precisions])
        elif isinstance(customized_precisions, str):
            self._precisions = sorted([i.strip() for i in customized_precisions.split(',')])
        self.conf.usr_cfg.mixed_precision.precisions = self._precisions

    @property
    def eval_dataloader(self):
        """Get eval_dataloader."""
        return self._eval_dataloader

    @eval_dataloader.setter
    def eval_dataloader(self, dataloader):
        """Set Dataloader for evaluation.

        It is iterable and the batched data should consists of a tuple like (input, label), 
        when eval_dataloader is set, user should configure postprocess(optional) and metric 
        in yaml file or set postprocess and metric cls. Notice evaluation dataloader will be 
        used to generate data for model inference, make sure the input data can be feed to model.

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
        assert hasattr(dataloader, '__iter__') and \
            hasattr(dataloader, 'batch_size'), \
            'dataloader must implement __iter__ method and batch_size attribute'

        self._eval_dataloader = dataloader

    @property
    def model(self):
        """Get model."""
        return self._model

    @model.setter
    def model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object.

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
        cfg = self.conf.usr_cfg
        if cfg.model.framework == 'NA':
            if isinstance(user_model, BaseModel):
                cfg.model.framework = list(MODELS.keys())[list(MODELS.values()).index(type(user_model))]
                if cfg.model.backend == "ipex":
                    assert cfg.model.framework == "pytorch_ipex", "Please wrap the model with correct Model class!"
                if cfg.model.backend == "itex":
                    from .model.tensorflow_model import get_model_type
                    if get_model_type(user_model.model) == 'keras':
                        assert cfg.model.framework == "keras", "Please wrap the model with KerasModel class!"
                    else:
                        assert cfg.model.framework == "pytorch_itex", \
                            "Please wrap the model with TensorflowModel class!"
            else:
                framework = get_model_fwk_name(user_model)
                if framework == "tensorflow":
                    from .model.tensorflow_model import get_model_type
                    if get_model_type(user_model) == 'keras' and cfg.model.backend == 'itex':
                        framework = 'keras'
                if framework == "pytorch":
                    if cfg.model.backend == "default":
                        framework = "pytorch_fx"
                    elif cfg.model.backend == "ipex":
                        framework = "pytorch_ipex"
                cfg.model.framework = framework

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
    def metric(self):
        """Get metric."""
        assert False, 'Should not try to get the value of `metric` attribute.'

    @metric.setter
    def metric(self, user_metric):
        """Set metric class or a dict of built-in metric configures.

        1. neural_compressor have many built-in metrics, user can pass a metric configure dict to tell neural 
           compressor what metric will be use.
           You can set multi-metrics to evaluate the performance of a specific model.
                Single metric:
                    {topk: 1}

                Multi-metrics:
                    {topk: 1,
                     MSE: {compare_label: False},
                    }
            Refer to this [file](../docs/source/metric.md#supported-built-in-metric-matrix) for built-in metric list
        2. User also can set specific metric through this api. The metric class should take the outputs of the model or
           postprocess(if have) as inputs, neural_compressor built-in metric always take(predictions, labels) as inputs
           for update, and user_metric.metric_cls should be sub_class of neural_compressor.metric.BaseMetric.

        Args:
            user_metric(neural_compressor.metric.Metric or a dict of built-in metric configures):
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
    def eval_func(self):
        """Get evaluation function."""
        assert False, 'Should not try to get the value of `eval_func` attribute.'

    @eval_func.setter
    def eval_func(self, user_eval_func):
        """Set evaluation function provided by user.

        Args:
            user_eval_func: This function takes "model" as input parameter
                            and executes entire evaluation process with self
                            contained metrics. If eval_func set,
                            an evaluation process must be triggered
                            to make evaluation of the model executed.
        """
        self._eval_func = user_eval_func

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
        config (MixedPrecisionConfig):        The path to the YAML configuration file or
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
        A _MixedPrecision object that generates low precision model across various DL frameworks.

    Raises:
        AssertionError.

    Example::

        from neural_compressor import mix_precision
        from neural_compressor.config import MixedPrecisionConfig

        conf = MixedPrecisionConfig()
        converted_model = mix_precision.fit(model, config=conf)
    """
    converter = _MixedPrecision(config)
    if config.precision in config.excluded_precisions:
        logger.warning("Target precision is in excluded_precisions, "\
            "please modify precision or excluded_precisions to make it understandable.")
        sys.exit(0)
    precisions = list(set(config.precision) - set(config.excluded_precisions))
    converter.precisions = precisions
    converter.model = model

    if ('bf16' in precisions or 'fp16' in precisions) and converter.model.framework() == "onnxruntime":
        if config.device == "cpu":
            logger.warning("Mix precision exits due to device isn't gpu for onnx models.")
            sys.exit(0)
        elif config.backend != "onnxrt_cuda_ep":
            logger.warning("Mix precision exits due to backend isn't onnxrt_cuda_ep for onnx models.")
            sys.exit(0)
    elif 'bf16' in precisions and not CpuInfo().bf16 and converter.model.framework() != "onnxruntime":
        if os.getenv('FORCE_BF16') == '1':
            logger.warning("Mix precision will generate bf16 graph although " \
                           "the hardware doesn't support bf16 instruction.")
        else:
            logger.warning("Mix precision exits due to the hardware " \
                           "doesn't support bf16 instruction.")
            sys.exit(0)
    elif 'fp16' in precisions and converter.model.framework() != "onnxruntime":
        logger.warning("Currently mix precision only supports fp16 for onnx models.")
        sys.exit(0)
    if eval_func is not None:
        converter.eval_func = eval_func
    if eval_dataloader is not None:
        converter.eval_dataloader = eval_dataloader
    if eval_metric is not None:
        converter.metric = eval_metric
    return converter()
