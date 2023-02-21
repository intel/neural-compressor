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

"""This is a module for Component class.

The Component class will be inherited by the class 'Quantization', 'Pruning' and 'Distillation'.
"""

import numpy as np
import os
import pickle
import random
from ..adaptor import FRAMEWORKS
from ..conf.config import QuantConf
from ..conf.dotdict import deep_get, deep_set, DotDict
from ..conf.pythonic_config import Config
from ..utils import logger
from ..utils.utility import time_limit
from ..model import BaseModel, Model
from ..model.model import get_model_fwk_name
from ..model.tensorflow_model import TensorflowQATModel
from ..strategy import STRATEGIES


class BaseCallbacks(object):
    """This is base class of Neural Compressor Callbacks.

    This class will be inherited by the class 'QuantizationCallbacks', 'PruningCallbacks' and 'DistillationCallbacks'.
    This design is mainly for pruning/distillation/quantization-aware training.
    In this class will apply all hooks for 'Quantization', 'Pruning' and 'Distillation'.
    """

    def __init__(self, conf=None, model=None):
        """Construct all the necessary attributes for the callbacks object.

        Args:
            conf: A Config object which definds the compressor behavior.
                  Just like:QuantizationAwareTrainingConfig, WeightPruningConfig and DistillationConfig.
            model: Model to be compressed in this object.
        """
        self.conf = None
        self.cfg = None
        self.framework = None
        self._model = None
        self.model = model
        self._train_func = None
        self._train_dataloader = None
        self._eval_func = None
        self._eval_dataloader = None
        self._train_distributed = False
        self._evaluation_distributed = False
        self.adaptor = None
        self._metric = None
        self.hooks = {
            'on_train_begin': self.on_train_begin,
            'on_train_end': self.on_train_end,
            'on_epoch_begin': self.on_epoch_begin,
            'on_epoch_end': self.on_epoch_end,
            'on_step_begin': self.on_step_begin,
            'on_step_end': self.on_step_end,
            'on_after_compute_loss': self.on_after_compute_loss,
            'on_before_optimizer_step': self.on_before_optimizer_step,
            'on_after_optimizer_step': self.on_after_optimizer_step,
            'on_before_eval': self.on_before_eval,
            'on_after_eval': self.on_after_eval
        }
        self.hooks_dict = {
            'on_train_begin': [],
            'on_train_end': [],
            'on_epoch_begin': [],
            'on_epoch_end': [],
            'on_step_begin': [],
            'on_step_end': [],
            'on_after_compute_loss': [],
            'on_before_optimizer_step': [],
            'on_after_optimizer_step': [],
            'on_before_eval': [],
            'on_after_eval': []
        }

    def on_train_begin(self, dataloader=None):
        """Be called before the beginning of epochs."""
        for on_train_begin_hook in self.hooks_dict['on_train_begin']:
            on_train_begin_hook(dataloader)

    def on_train_end(self):
        """Be called after the end of epochs."""
        for on_train_end_hook in self.hooks_dict['on_train_end']:
            on_train_end_hook()

    def on_epoch_begin(self, epoch):
        """Be called on the beginning of epochs."""
        for on_epoch_begin_hook in self.hooks_dict['on_epoch_begin']:
            on_epoch_begin_hook(epoch)

    def on_step_begin(self, batch_id):
        """Be called on the beginning of batches."""
        res_list = []
        for on_step_begin_hook in self.hooks_dict['on_step_begin']:
            res_list.append(on_step_begin_hook(batch_id))
        return res_list

    def on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None):
        """Be called on the end of loss computation."""
        loss = student_loss
        for on_after_compute_loss_hook in self.hooks_dict['on_after_compute_loss']:
            loss = on_after_compute_loss_hook(input, student_output, loss, teacher_output)
        return loss

    def on_before_optimizer_step(self):
        """Be called before optimizer step."""
        for on_before_optimizer_step_hook in self.hooks_dict['on_before_optimizer_step']:
            on_before_optimizer_step_hook()

    def on_after_optimizer_step(self):
        """Be called after optimizer step."""
        for on_after_optimizer_step_hook in self.hooks_dict['on_after_optimizer_step']:
            on_after_optimizer_step_hook()

    def on_before_eval(self):
        """Be called before evaluation."""
        for on_before_eval_hook in self.hooks_dict['on_before_eval']:
            on_before_eval_hook()

    def on_after_eval(self):
        """Be called after evaluation."""
        for on_after_eval_hook in self.hooks_dict['on_after_eval']:
            on_after_eval_hook()

    def on_step_end(self):
        """Be called on the end of batches."""
        res_list = []
        for on_step_end_hook in self.hooks_dict['on_step_end']:
            res_list.append(on_step_end_hook())
        return res_list

    def on_epoch_end(self):
        """Be called on the end of epochs."""
        res_list = []

        for on_epoch_end_hook in self.hooks_dict['on_epoch_end']:
            res_list.append(on_epoch_end_hook())

        return res_list

    def register_hook(self, scope, hook, input_args=None, input_kwargs=None):
        """Register hook for component.

        Input_args and input_kwargs are reserved for user registered hooks.
        """
        if hook not in self.hooks_dict[scope]:
            self.hooks_dict[scope].append(hook)

    def __repr__(self):
        """Represent this class."""
        pass

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
                        neural_compressor.experimental.common.Model.
                        If tensorflow model is used, model's inputs/outputs will be
                        auto inferenced, but sometimes auto inferenced
                        inputs/outputs will not meet your requests,
                        set them manually in config yaml file.
                        Another corner case is slim model of tensorflow,
                        be careful of the name of model configured in yaml file,
                        make sure the name is in supported slim model list.

        """
        if user_model is None:
            return

        if self.cfg.model.framework == 'NA':
            self.framework = get_model_fwk_name(
                user_model.model if isinstance(user_model, BaseModel) else user_model)
            if self.framework == "tensorflow":
                from ..model.tensorflow_model import get_model_type
                if get_model_type(user_model) == 'keras' and self.cfg.model.backend == 'itex':
                    self.framework = 'keras'
            if self.framework == "pytorch":
                if self.cfg.model.backend == "default":
                    self.framework = "pytorch_fx"
                elif self.cfg.model.backend == "ipex":
                    self.framework = "pytorch_ipex"
            self.cfg.model.framework = self.framework

        if not isinstance(user_model, BaseModel):
            logger.warning("Force convert framework model to neural_compressor model.")
            if self.framework == 'tensorflow':
                if type(user_model) == str:
                    self._model = TensorflowQATModel(user_model)
                else:
                    self._model = TensorflowQATModel(user_model._model)
            elif "tensorflow" in self.framework or self.framework == "keras":
                self._model = Model(user_model, backend=self.framework, device=self.cfg.device)
            else:
                self._model = Model(user_model, backend=self.framework)
        else:
            self._model = user_model

        if 'tensorflow' in self.framework:
            self._model.name = self.cfg.model.name
            self._model.output_tensor_names = self.cfg.model.outputs
            self._model.input_tensor_names = self.cfg.model.inputs
            self._model.workspace_path = self.cfg.tuning.workspace.path


class AwareTrainingQuantCallbacks(BaseCallbacks):
    """This is the class for callbacks of quantization aware training.

    This design is mainly for Quantization-Aware Training.
    In this class will apply all hooks for Quantization-Aware Training.
    """

    def __init__(self, conf=None, model=None):
        """Construct all the necessary attributes for the callbacks object.

        Args:
            conf: A QuantizationAwareTrainingConfig object which definds the compressor behavior.
            model: Model to be quantized in this object.
        """
        super(AwareTrainingQuantCallbacks, self).__init__(conf=None)
        conf = Config(quantization=conf, benchmark=None, pruning=None, distillation=None, nas=None)
        self.conf = QuantConf()
        self.conf.map_pyconfig_to_cfg(conf)
        self.cfg = self.conf.usr_cfg
        self.model = model

        seed = self.conf.usr_cfg.tuning.random_seed
        random.seed(seed)
        np.random.seed(seed)

        framework_specific_info = {'device': self.cfg.device,
                                   'random_seed': self.cfg.tuning.random_seed,
                                   'workspace_path': self.cfg.tuning.workspace.path,
                                   'q_dataloader': None,
                                   'backend': self.cfg.model.get('backend', 'default'),
                                   'format': self.cfg.model.get('quant_format', 'default'),
                                   'performance_only': self.cfg.model.get('tuning.exit_policy.performance_only', False)}
        if self.cfg.quantization.approach is not None:
            framework_specific_info['approach'] = self.cfg.quantization.approach

        if 'tensorflow' in self.framework:
            framework_specific_info.update(
                {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})
        self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)
        self.adaptor.model = self.model
        self.register_hook('on_train_begin', self.adaptor._pre_hook_for_qat)
        self.register_hook('on_train_end', self.adaptor._post_hook_for_qat)

    def pre_process(self):
        """Create strategy to optimize model."""
        # Remove qat hooks if user want to tune accuracy with train function.
        if hasattr(self.adaptor, "_pre_hook_for_qat"):
            self.remove_hook("on_train_begin", self.adaptor._pre_hook_for_qat)
            self.remove_hook("on_train_end", self.adaptor._post_hook_for_qat)

        strategy = self.cfg.tuning.strategy.name.lower()
        if self.cfg.quantization.quant_level == 0:
            strategy = "conservative"
            logger.info(f"On the premise that the accuracy meets the conditions, improve the performance.")

        if strategy == "mse_v2":
            if not (self.cfg.model.framework.startswith("tensorflow") or self.cfg.model.framework == 'pytorch_fx'):
                strategy = "basic"
                logger.warning(f"MSE_v2 does not support {self.cfg.model.framework} now, use basic instead.")
                logger.warning("Only tensorflow, pytorch_fx is supported by MSE_v2 currently.")
        assert strategy in STRATEGIES, "Tuning strategy {} is NOT supported".format(strategy)

        _resume = None
        # check if interrupted tuning procedure exists. if yes, it will resume the
        # whole auto tune process.
        self.resume_file = os.path.abspath(os.path.expanduser(self.cfg.tuning.workspace.resume)) \
                           if self.cfg.tuning.workspace and self.cfg.tuning.workspace.resume else None
        if self.resume_file:
            assert os.path.exists(self.resume_file), \
                "The specified resume file {} doesn't exist!".format(self.resume_file)
            with open(self.resume_file, 'rb') as f:
                _resume = pickle.load(f).__dict__

        self.strategy = STRATEGIES[strategy](
            self._model,
            self.conf,
            None,
            self._train_func,
            self._eval_dataloader,
            self._eval_func,
            _resume,
            None)

    def execute(self):
        """Quantization Aware Training execute routinue based on strategy design."""
        try:
            with time_limit(self.conf.usr_cfg.tuning.exit_policy.timeout):
                logger.debug("Dump user yaml configuration:")
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

        For derived classes, an override function is required.
        """
        self.pre_process()
        results = self.execute()
        return results

    fit = __call__

    def __repr__(self):
        """Represent this class."""
        return "Quantization Aware Training"

    def remove_hook(self, scope, hook):
        """Remove hooks if user want to tune accuracy with train_func."""
        for registed_hook in self.hooks_dict[scope]:
            if type(hook) == type(registed_hook):
                self.hooks_dict[scope].remove(registed_hook)

    @property
    def train_func(self):
        """Not support get train_func."""
        assert False, 'Should not try to get the value of `train_func` attribute.'
        return None

    @train_func.setter
    def train_func(self, user_train_func):
        """Training function.

        Args:
            user_train_func: This function takes "model" as input parameter
                         and executes entire training process with self
                         contained training hyper-parameters. If training_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func
                         to make evaluation of the model executed. training_func will return
                         a trained model.
        """
        self._train_func = user_train_func

    @property
    def eval_func(self):
        """Not support get eval_func."""
        assert False, 'Should not try to get the value of `eval_func` attribute.'
        return None

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
        User only need to set eval_dataloader when eval_dataloader can not be
        configured from yaml file.

        Args:
            dataloader(generator): user are supported to set a user defined dataloader
                                   which meet the requirements that can yield tuple of
                                   (input, label)/(input, _) batched data. Another good
                                   practice is to use neural_compressor.experimental.common.DataLoader
                                   to initialize a neural_compressor dataloader object. Notice
                                   neural_compressor.experimental.common.DataLoader is just a wrapper of the
                                   information needed to build a dataloader, it can't yield
                                   batched data and only in this setter method
                                   a 'real' train_dataloader will be created,
                                   the reason is we have to know the framework info
                                   and only after the Component object created then
                                   framework information can be known.
                                   Future we will support creating iterable dataloader
                                   from neural_compressor.experimental.common.DataLoader.
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
        """Set metric class or a dict of built-in metric configures,
        and neural_compressor will initialize this class when evaluation.

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
            user_metric(neural_compressor.metric.Metric or a dict of built-in metric configures):

        """
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.metric"):
            logger.warning("Override the value of `metric` field defined in yaml file" \
                           " as user defines the value of `metric` attribute by code.")

        from ..metric import Metric as NCMetric, METRICS
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


class PruningCallbacks(BaseCallbacks):
    pass


class DistillationCallbacks(BaseCallbacks):
    pass