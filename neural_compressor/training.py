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

from .conf.config import DistillationConf, PruningConf, QuantConf
from .experimental.common import Model
from .experimental.distillation import Distillation
from .experimental.pruning import Pruning
from .experimental.quantization import Quantization
from .experimental.scheduler import Scheduler
from .model.base_model import BaseModel
from .utils import logger
from typing import Callable, List, Union


class TrainingCallbacks:
    """TrainingCallbacks is uesd in train loop for what user want to deal with additional
    arguments:
        commponent: one instance of Distillation, Quantization, Pruning, Scheduler

    examples:
        import neural_compressor.training.prepare
        training_callbacks = prepare(conf, model)
        train_loop:
            training_callbacks.on_train_begin()
            for epoch in range(epochs):
                training_callbacks.on_epoch_begin(epoch)
                for i, batch in enumerate(dataloader):
                    training_callbacks.on_step_begin(i)
                    ......
                    output = model(batch)
                    loss = ......
                    loss = training_callbacks.on_after_compute_loss(batch, output, loss)
                    loss.backward()
                    training_callbacks.on_before_optimizer_step()
                    optimizer.step()
                    training_callbacks.on_step_end()
                training_callbacks.on_epoch_end()
            training_callbacks.on_train_end()
    """
    def __init__(self, component):
        self.component = component
        self._component = \
            component.components[0] if isinstance(component, Scheduler) else component

    def on_train_begin(self):
        """ called before the beginning of training"""
        for on_train_begin_hook in self._component.hooks_dict['on_train_begin']:
            on_train_begin_hook()

    def on_train_end(self):
        """ called after the end of epochs"""
        for on_train_end_hook in self._component.hooks_dict['on_train_end']:
            on_train_end_hook()

    def on_epoch_begin(self, epoch):
        """ called on the beginning of epochs"""
        for on_epoch_begin_hook in self._component.hooks_dict['on_epoch_begin']:
            on_epoch_begin_hook(epoch)

    def on_step_begin(self, batch_id):
        """ called on the beginning of batches"""
        res_list = []
        for on_step_begin_hook in self._component.hooks_dict['on_step_begin']:
            res_list.append(on_step_begin_hook(batch_id))
        return res_list

    def on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None):
        """ called on the end of loss computation"""
        loss = student_loss
        for on_after_compute_loss_hook in self._component.hooks_dict['on_after_compute_loss']:
            loss = on_after_compute_loss_hook(input, student_output, loss, teacher_output)
        return loss

    def on_before_optimizer_step(self):
        """ called on the end of backward"""
        for on_before_optimizer_step_hook in self._component.hooks_dict['on_before_optimizer_step']:
            on_before_optimizer_step_hook()

    def on_step_end(self):
        """ called on the end of batches"""
        res_list = []
        for on_step_end_hook in self._component.hooks_dict['on_step_end']:
            res_list.append(on_step_end_hook())
        return res_list

    def on_epoch_end(self):
        """ called on the end of epochs"""
        res_list = []

        for on_epoch_end_hook in self._component.hooks_dict['on_epoch_end']:
            res_list.append(on_epoch_end_hook())

        return res_list


def prepare(
    confs: Union[Callable, List], model: Callable = None, teacher_model: Callable = None,
    features=None
):
    """_summary_

    Args:
        confs (Union[Callable, List]): config of Distillation, Quantization, Pruning,
                                       or list of config for orchestration optimization
        model (Callable, optional): model to optimize. Defaults to None.
        teacher_model (Callable, optional): teacher model for distillation. Defaults to None.
        features (optional): teacher features for distillation, features and teacher_model are alternative.
                                     Defaults to None.

    Returns:
        TrainingCallbacks
        model

    examples:
        import neural_compressor.training.prepare
        training_callbacks = prepare(conf, model)
        train_loop:
            training_callbacks.on_train_begin()
            for epoch in range(epochs):
                training_callbacks.on_epoch_begin(epoch)
                for i, batch in enumerate(dataloader):
                    training_callbacks.on_step_begin(i)
                    ......
                    output = model(batch)
                    loss = ......
                    loss = training_callbacks.on_after_compute_loss(batch, output, loss)
                    loss.backward()
                    training_callbacks.on_before_optimizer_step()
                    optimizer.step()
                    training_callbacks.on_step_end()
                training_callbacks.on_epoch_end()
            training_callbacks.on_train_end()
    """
    if isinstance(confs, List):
        from .experimental.scheduler import Scheduler
        comps = []
        for config in confs:
            if isinstance(config, QuantConf):
                com = Quantization(config)
            elif isinstance(config, PruningConf):
                com = Pruning(config)
            elif isinstance(config, DistillationConf):
                com = Distillation(config)
                if teacher_model is not None:
                    com.teacher_model = teacher_model
            else:
                assert False, "Unsupported component: {}".format(config)

            comps.append(com)
        scheduler = Scheduler()
        assert model is not None, "need set neural_compressor Model for scheduler."
        scheduler.model = model
        comp = scheduler.combine(*comps)
        scheduler.append(comp)
        component = scheduler
    else:
        if isinstance(confs, QuantConf):
            component = Quantization(confs)
        elif isinstance(confs, PruningConf):
            component = Pruning(confs)
        elif isinstance(confs, DistillationConf):
            component = Distillation(confs)
            if teacher_model is not None:
                component.teacher_model = teacher_model
        else:
            assert False, logger.error(
                "confs should be one of QuantConf, PruningConf, DistillationConf. not ", confs
            )

        if model is not None:
            component.model = Model(model)

    return TrainingCallbacks(component), component.model


def fit(
    model: BaseModel,
    callbacks: TrainingCallbacks,
    train_func: Callable,
    eval_func: Callable
):
    """_summary_

    Args:
        model: neural_compressor.model.base_model.BaseModel, which to be optimized
        callbacks (TrainingCallbacks): instance of TrainingCallbacks which from prepare function
        train_func (function, optional): _description_. Defaults to None.
        eval_func (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    examples:
        import neural_compressor.training.prepare
        training_callbacks = prepare(conf, model)
        def train_func(model):
            training_callbacks.on_train_begin()
            for epoch in range(epochs):
                training_callbacks.on_epoch_begin(epoch)
                for i, batch in enumerate(dataloader):
                    training_callbacks.on_step_begin(i)
                    ......
                    output = model(batch)
                    loss = ......
                    loss = training_callbacks.on_after_compute_loss(batch, output, loss)
                    loss.backward()
                    training_callbacks.on_before_optimizer_step()
                    optimizer.step()
                    training_callbacks.on_step_end()
                training_callbacks.on_epoch_end()
            training_callbacks.on_train_end()
            return model

        def eval_func(model):
            ......

        fit(model, training_callbacks, train_func, eval_func)
    """
    callbacks.component.model = model
    assert train_func is not None, "train_func must be set!"
    assert eval_func is not None, "eval_func must be set!"
    if isinstance(callbacks.component, Scheduler):
        callbacks.component.components[0].train_func = train_func
    else:
        callbacks.component.train_func = train_func
    if isinstance(callbacks.component, Scheduler):
        callbacks.component.components[0].eval_func = eval_func
    else:
        callbacks.component.eval_func = eval_func

    return callbacks.component()
