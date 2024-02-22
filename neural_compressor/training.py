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
import os
import pickle
import random
from typing import Callable, List, Union

import numpy as np

from neural_compressor import DistillationConfig, QuantizationAwareTrainingConfig, WeightPruningConfig
from neural_compressor.strategy.strategy import STRATEGIES

from .adaptor import FRAMEWORKS
from .compression.callbacks import DistillationCallbacks, PruningCallbacks, QuantizationAwareTrainingCallbacks
from .compression.pruner import prepare_pruning
from .config import _Config, options
from .metric import register_customer_metric
from .model.model import Model
from .utils import logger
from .utils.utility import time_limit


class CompressionManager:
    """CompressionManager is used in train loop for what user want to deal with additional.

    Arguments:
        model: A model to be compressed.
        confs: The instance of QuantizationAwareTrainingConfig, PruningConfig and distillationConfig, or a list of
               config for orchestration optimization.

    Examples::

        import neural_compressor.training.prepare_compression
        compression_manager = prepare_compression(model, confs)
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model
        # train_loop:
        for epoch in range(epochs):
            compression_manager.callbacks.on_epoch_begin(epoch)
            for i, (batch, label) in enumerate(dataloader):
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

    def __init__(self, model: Callable, confs: Union[Callable, List], **kwargs):
        """Initialize the CompressionManager's parameters.

        model: A model to be compressed.
        confs: The instance of QuantizationAwareTrainingConfig, PruningConfig and distillationConfig, or a list of
               config for orchestration optimization.
        """
        callbacks_list = []
        self.model = None
        q_conf = None
        p_conf = None
        d_conf = None
        self.adaptor = None

        if isinstance(confs, List) and len(confs) > 1:
            for conf in confs:
                if isinstance(conf, QuantizationAwareTrainingConfig) or isinstance(conf, WeightPruningConfig):
                    self.model = Model(model, conf=conf)
            if self.model is None:
                self.model = Model(model)

            for conf in confs:
                if isinstance(conf, QuantizationAwareTrainingConfig):
                    q_conf = conf

                    framework_specific_info = {
                        "device": conf.device,
                        "random_seed": options.random_seed,
                        "workspace_path": options.workspace,
                        "q_dataloader": None,
                        "backend": getattr(confs, "backend", "default"),
                        "format": getattr(confs, "quant_format", "default"),
                        "approach": conf.approach,
                    }
                    if "tensorflow" in conf.framework:
                        framework_specific_info.update({"inputs": conf.inputs, "outputs": conf.outputs})

                    # TODO: will be removed once 'op_type_dict' and 'op_name_dicts'
                    # for quant_aware_training can be handled in strategy
                    framework_specific_info["qat_optype_wise"] = conf.op_type_dict
                    framework_specific_info["qat_op_wise"] = conf.op_name_dict

                    self.adaptor = FRAMEWORKS[conf.framework](framework_specific_info)
                    self.adaptor.model = self.model
                    callbacks_list.append(QuantizationAwareTrainingCallbacks(conf, adaptor=self.adaptor))
                elif isinstance(conf, WeightPruningConfig):
                    p_conf = conf
                    callbacks_list.append(PruningCallbacks(conf, model=self.model))
                elif isinstance(conf, DistillationConfig):
                    d_conf = conf
                    callbacks_list.append(DistillationCallbacks(conf, model=self.model))
                else:
                    assert False, "Unsupported configure: {}".format(type(conf))
            self.conf = _Config(quantization=q_conf, benchmark=None, pruning=p_conf, distillation=d_conf, nas=None)
        else:
            if isinstance(confs, List):
                confs = confs[0]
            if isinstance(confs, QuantizationAwareTrainingConfig):
                self.model = Model(model, conf=confs)

                framework_specific_info = {
                    "device": confs.device,
                    "random_seed": options.random_seed,
                    "workspace_path": options.workspace,
                    "q_dataloader": None,
                    "backend": getattr(confs, "backend", "default"),
                    "format": getattr(confs, "quant_format", "default"),
                    "approach": confs.approach,
                }
                if "tensorflow" in confs.framework:
                    framework_specific_info.update({"inputs": confs.inputs, "outputs": confs.outputs})

                # TODO: will be removed once 'op_type_dict' and 'op_name_dicts'
                # for quant_aware_training can be handled in strategy
                framework_specific_info["qat_optype_wise"] = confs.op_type_dict
                framework_specific_info["qat_op_wise"] = confs.op_name_dict

                self.adaptor = FRAMEWORKS[confs.framework](framework_specific_info)
                self.adaptor.model = self.model
                callbacks_list.append(QuantizationAwareTrainingCallbacks(confs, adaptor=self.adaptor))
                self.conf = _Config(quantization=confs, benchmark=None, pruning=None, distillation=None, nas=None)
            elif isinstance(confs, WeightPruningConfig):
                self.model = Model(model, conf=confs)
                callbacks_list.append(PruningCallbacks(confs, model=self.model))
                self.conf = _Config(quantization=None, benchmark=None, pruning=confs, distillation=None, nas=None)
            elif isinstance(confs, DistillationConfig):
                self.model = Model(model)
                callbacks_list.append(DistillationCallbacks(confs, model=self.model))
                self.conf = _Config(quantization=None, benchmark=None, pruning=None, distillation=confs, nas=None)
            else:
                assert False, logger.error(
                    "confs should be one of QuantizationAwareTrainingConfig, "
                    "PruningConfig, DistillationConfig. not {}".format(type(confs))
                )

        try:
            # TODO: export to ONNX model need original fp32 model now, will remove it
            #  when int8 model can be exported to ONNX model.
            self.fp32_model = self.model
        except Exception as e:  # pragma: no cover
            logger.warning("Fail to deep copy the model due to {}.".format(repr(e)))
            self.fp32_model = None

        self.callbacks = CallBacks(callbacks_list)

    def save(self, root=None):
        """Save compressed model.

        Args:
            root (str): path to save the model
        """
        self.model.save(root)  # pylint: disable=no-member

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
        self.model.export(save_path, conf)  # pylint: disable=no-member


def fit(compression_manager, train_func, eval_func=None, eval_dataloader=None, eval_metric=None, **kwargs):
    """Compress the model with accuracy tuning for quantization.

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
                                              evaluation metrics object and should set "eval_func" parameter as None.
                                              Tuner will combine model, eval_dataloader
                                              and pre-defined metrics to run evaluation
                                              process.
        eval_metric (dict or obj):            Set metric class or a dict of built-in metric configures,
                                              and neural_compressor will initialize this class when evaluation.

    Returns:
        A optimized model.

    Examples::

        from neural_compressor.training import fit, prepare_compression

        compression_manager = prepare_compression(conf, model)

        def train_func(model):
            compression_manager.callbacks.on_train_begin()
            for epoch in range(epochs):
                compression_manager.callbacks.on_epoch_begin(epoch)
                for i, (batch, label) in enumerate(dataloader):
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
            return model

        def eval_func(model):
            for i, (batch, label) in enumerate(dataloader):
                output = model(batch)
                # compute metric
                metric = top1(output, label)
            return metric.results()

        model = fit(compression_manager, train_func=train_func, eval_func=eval_func)
    """
    assert compression_manager.conf.quantization is not None, "Only quantization supports tuning with accuracy driven."
    seed = options.random_seed
    random.seed(seed)
    np.random.seed(seed)

    # Remove qat hooks if user want to tune accuracy with train function.
    for callback in compression_manager.callbacks.callbacks_list:
        if isinstance(callback, QuantizationAwareTrainingCallbacks):
            callback.remove_hook("on_train_begin", compression_manager.adaptor._pre_hook_for_qat)
            callback.remove_hook("on_train_end", compression_manager.adaptor._post_hook_for_qat)

    if eval_metric is not None:
        metric = register_customer_metric(eval_metric, compression_manager.conf.quantization.framework)
    else:
        metric = None

    strategy_name = compression_manager.conf.quantization.tuning_criterion.strategy

    if compression_manager.conf.quantization.quant_level == "auto":
        strategy_name = "auto"
    elif compression_manager.conf.quantization.quant_level == 0:
        strategy_name = "conservative"

    if strategy_name == "mse_v2":
        if not (
            compression_manager.conf.quantization.framework.startswith("tensorflow")
            or compression_manager.conf.quantization.framework == "pytorch_fx"
        ):  # pragma: no cover
            strategy_name = "basic"
            logger.warning(
                f"MSE_v2 does not support {compression_manager.conf.quantization.framework} now," "use basic instead."
            )
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
        model=compression_manager.model,
        conf=compression_manager.conf,
        q_dataloader=None,
        q_func=train_func,
        eval_func=eval_func,
        eval_dataloader=eval_dataloader,
        eval_metric=metric,
        resume=_resume,
        q_hooks=None,
    )
    try:
        with time_limit(compression_manager.conf.quantization.tuning_criterion.timeout):
            logger.debug("Dump user yaml configuration:")
            logger.debug(compression_manager.conf)
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

        compression_manager.model = strategy.get_best_qmodel()

    return compression_manager.model


def prepare_compression(model: Callable, confs: Union[Callable, List], **kwargs):
    """Summary.

    Args:
        model (Callable, optional):    The model to optimize.
        confs (Union[Callable, List]): The instance of QuantizationAwareTrainingConfig,
                                       PruningConfig and distillationConfig, or a list of
                                       config for orchestration optimization.

    Returns:
        An object of CompressionManager.

    Examples::

        from neural_compressor.training import prepare_compression

        compression_manager = prepare_compression(conf, model)
        model = compression_manager.model
        # train_loop:
        compression_manager.callbacks.on_train_begin()
        for epoch in range(epochs):
            compression_manager.callbacks.on_epoch_begin(epoch)
            for i, (batch, label) in enumerate(dataloader):
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
    """
    compression_manager = CompressionManager(model, confs, **kwargs)

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
