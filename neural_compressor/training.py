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
"""The configuration of the training loop."""
import copy
from .compression.callbacks import QuantizationAwareTrainingCallbacks, DistillationCallbacks, PruningCallbacks
from .model.model import Model
from .utils import logger
from neural_compressor import (DistillationConfig, QuantizationAwareTrainingConfig,
                               WeightPruningConfig)
from typing import Callable, List, Union
from .compression import prepare_pruning


class CompressionManager:
    """CompressionManager is uesd in train loop for what user want to deal with additional.

    Arguments:
        model: A model to be compressed. It should be neural compressor model.
        callbacks: A list of Callbacks instances.
                   Such as: DistillationCallbbacks, QuantizationAwareTrainingCallbacks, PruningCallbacks.

    Examples::

        import neural_compressor.training.prepare_compression
        compression_manager = prepare_compression(nc_model, confs)
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model
        train_loop:
            for epoch in range(epochs):
                compression_manager.callbacks.on_epoch_begin(epoch)
                for i, batch in enumerate(dataloader):
                    compression_manager.callbacks.on_step_begin(i)
                    ......
                    output = model(batch)
                    loss = ......
                    loss = compression_manager.callbacks.on_after_compute_loss(batch, output, loss)
                    loss.backward()
                    compression_manager.callbacks.on_before_optimizer_step()
                    optimizer.step()
                    compression_manager.callbacks.on_step_end()
                compression_manager.callbacks.on_epoch_end()
        compression_manager.callbacks.on_train_end()
        compression_manager.save("path_to_save")
    """
    def __init__(self, model, callbacks_list):
        """Initialize the CompressionManager's parameters.

        model: A model to be compressed. It should be neural compressor model.
        callbacks: A list of Callbacks instances.
                   Such as: DistillationCallbbacks, QuantizationAwareTrainingCallbacks, PruningCallbacks.
        """
        self.callbacks = CallBacks(callbacks_list)
        self.model = model
        self._train_func = None
        self._eval_func = None
        self.quantizer = None

        try:
            # TODO: export to ONNX model need original fp32 model now, will remove it
            #  when int8 model can be exported to ONNX model.
            self.fp32_model = model
        except Exception as e:  # pragma: no cover
            logger.warning("Fail to deep copy the model due to {}.".format(repr(e)))
            self.fp32_model = None

        for component in callbacks_list:
            if isinstance(component, QuantizationAwareTrainingCallbacks):
                self.quantizer = component

    @property
    def train_func(self):
        """Not support get train_func."""
        assert False, 'Should not try to get the value of `train_func` attribute.'

    @train_func.setter
    def train_func(self, user_train_func):
        """Set training function.

        Args:
            user_train_func: This function takes "model" as input parameter
                         and executes entire training process with self
                         contained training hyper-parameters. If training_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func
                         to make evaluation of the model executed. training_func will return
                         a trained model.
        """
        self.quantizer.train_func = user_train_func

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
        assert self.quantizer is not None, "There is no quantizer to tune, " \
                                           "please pass a QuantizationAwareTrainingConfig."
        self.quantizer.eval_func = user_eval_func

    @property
    def eval_dataloader(self):
        """Getter to eval dataloader."""
        return self.quantizer.eval_dataloader

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
        assert self.quantizer is not None, "There is no quantizer to tune, " \
                                           "please pass a QuantizationAwareTrainingConfig."
        assert hasattr(dataloader, '__iter__') and \
            hasattr(dataloader, 'batch_size'), \
            'dataloader must implement __iter__ method and batch_size attribute'

        self.quantizer.eval_dataloader = dataloader

    @property
    def metric(self):
        """Get `metric` attribute."""
        assert False, 'Should not try to get the value of `metric` attribute.'

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
        assert self.quantizer is not None, "There is no quantizer to tune, " \
                                           "please pass a QuantizationAwareTrainingConfig."
        self.quantizer.metric = user_metric

    def fit(self):
        """Compress model with tuning for quantization."""
        self.model = self.quantizer.fit()
        return self.model

    def save(self, root=None):
        """Save compressed model.

        Args:
            root (str): path to save the model
        """
        self.model.save(root)

    def export(
        self,
        save_path: str,
        conf,
    ):
        """Convert the model to another type model, like `onnx` model and so on.

        Args:
            save_path (str): The path to save the model
            conf (Union[Callable, List]) : The configure for onnx exportation.
        """
        self.model.export(save_path, conf)


def fit(compression_manager,
        train_func,
        eval_func=None,
        eval_dataloader=None,
        eval_metric=None,
        **kwargs):
    """Compress the model with tuning for quantization.

    Args:
        compression_manager (CompressionManager):  The Compression manager contains the model and
                                              callbacks.
        train_func (function, optional):      Training function for quantization aware training. It is optional.
                                              This function takes "model" as input parameter
                                              and executes entire inference process. If this
                                              parameter specified.
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
        eval_dataloader (generator, optional): Data loader for evaluation. It is iterable
                                              and should yield a tuple of (input, label).
                                              The input could be a object, list, tuple or
                                              dict, depending on user implementation,
                                              as well as it can be taken as model input.
                                              The label should be able to take as input of
                                              supported metrics. If this parameter is
                                              not None, user needs to specify pre-defined
                                              evaluation metrics object and should set "eval_func" paramter as None.
                                              Tuner will combine model, eval_dataloader
                                              and pre-defined metrics to run evaluation
                                              process.
        eval_metric (dict or obj):            Set metric class or a dict of built-in metric configures,
                                              and neural_compressor will initialize this class when evaluation.
    """
    assert compression_manager.quantizer is not None, "Only quantization supports tuning with accuracy driven."
    compression_manager.train_func = train_func
    if eval_func is not None:
        compression_manager.eval_func = eval_func
    if eval_dataloader is not None:
        compression_manager.eval_dataloader = eval_dataloader
    if eval_metric is not None:
        compression_manager.eval_metric = eval_metric
    return compression_manager.fit()


def prepare_compression(model: Callable, confs: Union[Callable, List], **kwargs):
    """Summary.

    Args:
        model (Callable, optional):    The model to optimize.
        confs (Union[Callable, List]): The instance of QuantizationAwareTrainingConfig,
                                       PruningConfig and distillationConfig, or a list of
                                       config for orchestration optimization.
        options (Options, optional):   The configure for random_seed, workspace,
                                       resume path and tensorboard flag.

    Returns:
        An object of CompressionManager.

    Examples::

        import neural_compressor.training.prepare_compression

        compression_manager = prepare_compression(conf, model)
        train_loop:
            compression_manager.on_train_begin()
            for epoch in range(epochs):
                compression_manager.on_epoch_begin(epoch)
                for i, batch in enumerate(dataloader):
                    compression_manager.on_step_begin(i)
                    ......
                    output = model(batch)
                    loss = ......
                    loss = compression_manager.on_after_compute_loss(batch, output, loss)
                    loss.backward()
                    compression_manager.on_before_optimizer_step()
                    optimizer.step()
                    compression_manager.on_step_end()
                compression_manager.on_epoch_end()
            compression_manager.on_train_end()
    """
    callbacks_list = []
    nc_model = None
    if isinstance(confs, List) and len(confs) > 1:
        for conf in confs:
            if isinstance(conf, QuantizationAwareTrainingConfig):
                nc_model = Model(model, backend=conf.backend, approach="quant_aware_training")
                callbacks_list.append(QuantizationAwareTrainingCallbacks(conf, model=nc_model))
            elif isinstance(conf, WeightPruningConfig):
                callbacks_list.append(PruningCallbacks(conf, model=model))
            elif isinstance(conf, DistillationConfig):
                callbacks_list.append(DistillationCallbacks(conf, model=model))
            else:
                assert False, "Unsupported configure: {}".format(type(conf))
    else:
        if isinstance(confs, List):
            confs = confs[0]
        if isinstance(confs, QuantizationAwareTrainingConfig):
            nc_model = Model(model, backend=confs.backend, approach="quant_aware_training")
            callbacks_list.append(QuantizationAwareTrainingCallbacks(confs, model=nc_model))
        elif isinstance(confs, WeightPruningConfig):
            callbacks_list.append(PruningCallbacks(confs, model=model))
        elif isinstance(confs, DistillationConfig):
            callbacks_list.append(DistillationCallbacks(confs, model=model))
        else:
            assert False, logger.error(
                "confs should be one of QuantizationAwareTrainingConfig, "
                "PruningConfig, DistillationConfig. not {}".format(type(confs))
            )

    if nc_model is None:
        nc_model = Model(model, backend="default")

    compression_manager = CompressionManager(nc_model, callbacks_list=callbacks_list)

    return compression_manager


class CallBacks:
    """Define the basic command for the training loop."""
    def __init__(self, callbacks_list):
        """Callbacks list are used for execute the training procedure.

        Callbacks list should be any of the instance of QuantizationAwareTrainingCallbacks,
        PruningCallbacks and DistillationCallbacks.
        """
        assert len(callbacks_list) >= 1, "The length of callbacks list must be greater than 1."
        self.callbacks_list = callbacks_list

    def on_train_begin(self, dataloader=None):
        """Be called before the beginning of training."""
        for callbacks in self.callbacks_list:
            callbacks.on_train_begin(dataloader)

    def on_train_end(self):
        """Be called after the end of training."""
        for callbacks in self.callbacks_list:
            callbacks.on_train_end()
        logger.info("Training finished!")

    def on_epoch_begin(self, epoch):
        """Be called on the beginning of epochs."""
        for callbacks in self.callbacks_list:
            callbacks.on_epoch_begin(epoch)

    def on_step_begin(self, batch_id):
        """Be called on the beginning of batches."""
        res_list = []
        for callbacks in self.callbacks_list:
            res = callbacks.on_step_begin(batch_id)
            if res is not None:
                res_list.append(res)
        return res_list

    def on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None):
        """Be called on the end of loss computation."""
        loss_list = []
        for callbacks in self.callbacks_list:
            loss = callbacks.on_after_compute_loss(input, student_output, student_loss, teacher_output)
            if loss is not None:
                loss_list.append(loss)
        return loss_list[0] if len(loss_list) == 1 else loss_list

    def on_before_optimizer_step(self):
        """Be called before optimizer step."""
        for callbacks in self.callbacks_list:
            callbacks.on_before_optimizer_step()

    def on_after_optimizer_step(self):
        """Be called after optimizer step."""
        for callbacks in self.callbacks_list:
            callbacks.on_after_optimizer_step()

    def on_before_eval(self):
        """Be called before evaluation."""
        for callbacks in self.callbacks_list:
            callbacks.on_before_eval()

    def on_after_eval(self):
        """Be called after evaluation."""
        for callbacks in self.callbacks_list:
            callbacks.on_after_eval()

    def on_step_end(self):
        """Be called on the end of batches."""
        res_list = []
        for callbacks in self.callbacks_list:
            res = callbacks.on_step_end()
            if res is not None:
                res_list.append(res)
        return res_list

    def on_epoch_end(self):
        """Be called on the end of epochs."""
        res_list = []
        for callbacks in self.callbacks_list:
            res_list.extend(callbacks.on_epoch_end())
        return res_list
