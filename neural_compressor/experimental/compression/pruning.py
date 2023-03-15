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

from neural_compressor.compression.pruner.utils import process_config, parse_to_prune
from neural_compressor.compression.pruner.pruners import get_pruner
from neural_compressor.compression.pruner.utils import logger, torch


def _generate_pruners(config, model):
    """Generate pruners.

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
    """Mount on_step_begin to the model.

    :param model:The torch module to be pruned
    :return: hook handle
    """

    def hook(module, input):
        for pruner in module.pruners:
            pruner.on_step_begin(0)

    hook_handle = model.register_forward_pre_hook(hook)
    return hook_handle


def _rewrite_optimizer_step(opt: torch.optim.Optimizer):
    """Mount on_before/after_optimizer_step to optimizer.

    :param opt: user optimizer
    :return: the modified optimizer
    """

    def new_step(self, closure=None):
        if hasattr(self, "pruners"):  ## in case user save the whole optimzer
            for pruner in self.pruners:
                pruner.on_before_optimizer_step()

        if closure is not None:
            res = self.orig_step(closure)
        else:
            res = self.orig_step()
        if hasattr(self, "pruners"):
            for pruner in self.pruners:
                pruner.on_after_optimizer_step()
        return res

    opt.orig_step = opt.step
    import types
    opt.step = types.MethodType(new_step, opt)
    return opt


def save(
        obj: object,
        f,
        pickle_module=None,
        pickle_protocol=None,
        _use_new_zipfile_serialization=None
):
    """A rewrite function for torch save.

    :param obj:
    :param f:
    :param pickle_module:
    :param pickle_protocol:
    :param _use_new_zipfile_serialization:
    :return:
    """
    params = {}
    if pickle_module != None:
        params['pickle_module'] = pickle_module
    if pickle_protocol != None:
        params['pickle_protocol'] = pickle_protocol
    if _use_new_zipfile_serialization != None:
        params['_use_new_zipfile_serialization'] = _use_new_zipfile_serialization

    if isinstance(obj, torch.nn.Module) and hasattr(obj, "pruners"):
        pruners = obj.pruners
        obj.pruners = None
        delattr(obj, "pruners")
        obj.inc_hook_handle.remove()
        delattr(obj, "inc_hook_handle")
        if len(params) != 0:
            torch.orig_save(obj, f, params)
        else:
            torch.orig_save(obj, f)
        ##recover
        obj.pruners = pruners
        inc_hook_handle = _register_on_step_begin(obj)
        obj.inc_hook_handle = inc_hook_handle
        return

    if isinstance(obj, torch.optim.Optimizer) and hasattr(obj, "orig_step"):
        pruners = obj.pruners
        obj.pruners = None
        delattr(obj, "pruners")
        obj.step = obj.orig_step
        delattr(obj, "orig_step")
        if len(params) != 0:
            torch.orig_save(obj, f, params)
        else:
            torch.orig_save(obj, f)
        ##recover
        _rewrite_optimizer_step(obj)
        obj.pruners = pruners
        return
    if len(params) != 0:
        torch.orig_save(obj, f, params)
    else:
        torch.orig_save(obj, f)


def prepare_pruning(config, model: torch.nn.Module, opt: torch.optim):
    """Wrapper the model and optimizer to support all the pruning functionality.

    :param config: WeightPruningConfig
    :param model: The user's model
    :param opt: The user's optimizer
    :return: The modified model and optimizer
    """
    import torch
    torch.orig_save = torch.save  ##rewrite torch save
    setattr(torch, 'save', save)

    pruners = _generate_pruners(config, model)
    model.pruners = pruners
    opt.pruners = pruners

    inc_hook_handle = _register_on_step_begin(model)
    model.inc_hook_handle = inc_hook_handle
    _rewrite_optimizer_step(opt)
    return model, opt

# def complete_pruning(model: torch.nn.Module, opt: torch.optim):
#     """UnWrapper the model and optimizer
#     :param model: the modified model
#     :param opt: the modified optimizer
#     :return: the pruned model and the user's optimizer
#     """
#
#     model.inc_hook_handle.remove()
#     delattr(model, "inc_hook_handle")
#     model.pruners = None
#     delattr(model, "pruners")
#     opt.pruners = None
#     delattr(opt, "pruners")
#     opt.step = opt.orig_step
#     delattr(opt, "orig_step")
#     return model, opt
