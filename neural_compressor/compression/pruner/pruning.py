"""Pruning."""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

from .utils import process_config, parse_to_prune
from .pruners import get_pruner
from .utils import logger, torch


def _generate_pruners(config, model):
    """Generate pruners
    :param config: WeightPruningConfig
    :param model: The torch module to be pruned
    :return: A list of pruner
    """
    assert isinstance(model, torch.nn.Module)
    pruners_info = process_config(config)
    pruners = []
    for info in pruners_info:
        modules = parse_to_prune(info, model)
        if modules == {}:
            logger.warning("one pruner hooks no layers, please have a check")

        pruners.append(get_pruner(info, modules))
        info['modules'] = [key for key in modules.keys()]
        info['len_of_modules'] = len(info['modules'])
        logger.info(info)
    return pruners


def _register_on_step_begin(model):
    """mount on_step_begin to the model
    :param model:The torch module to be pruned
    :return: hook handle
    """
    def hook(module, input):
        for pruner in module.pruners:
            pruner.on_step_begin(0)

    hook_handle = model.register_forward_pre_hook(hook)
    return hook_handle


def rewrite_optimizer_step(opt: torch.optim.Optimizer):
    """mount on_before/after_optimizer_step to optimizer
    :param opt: user optimizer
    :return: the modified optimizer
    """
    def new_step(self, closure=None):
        for pruner in self.pruners:
            pruner.on_before_optimizer_step()

        if closure is not None:
            res = self.orig_step(closure)
        else:
            res = self.orig_step()
        for pruner in self.pruners:
            pruner.on_after_optimizer_step()
        return res

    opt.orig_step = opt.step
    import types
    opt.step = types.MethodType(new_step, opt)
    return opt


def PruningWrapper(config, model: torch.nn.Module, opt: torch.optim):
    """
    Wrapper to model and optimizer to support all the pruning functionality
    :param config: WeightPruningConfig
    :param model: The user's model
    :param opt: The user's optimizer
    :return: The modified model and optimizer
    """
    pruners = _generate_pruners(config, model)  ##list
    model.pruners = pruners
    opt.pruners = pruners

    inc_hook_handle = _register_on_step_begin(model)
    model.inc_hook_handle = inc_hook_handle
    rewrite_optimizer_step(opt)
    return model, opt


def PruningUnWrapper(model: torch.nn.Module, opt: torch.optim):
    """
    :param model: the modified model
    :param opt: the modified optimizer
    :return: the pruned model and the user's optimizer
    """
    model.inc_hook_handle.remove()
    delattr(model, "inc_hook_handle")
    opt.step = opt.orig_step
    delattr(opt, "orig_step")
    return model, opt
