"""Pruning init."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

# model slim related
from .model_slim.auto_slim import parse_auto_slim_config
from .model_slim.auto_slim import model_slim
from .pruning import PRUNINGS
from .utils import process_config, torch, logger
from typing import Optional, Union

FRAMEWORK = {"pytorch": "pt", "keras": "keras"}


def _register_on_step_begin(model):
    """Mount on_step_begin to the model.

    :param model:The torch module to be pruned
    :return: hook handle
    """

    def hook(module, input):
        for pruning in module.prunings:
            pruning.on_step_begin()

    hook_handle = model.register_forward_pre_hook(hook)
    return hook_handle


# def _register_on_step_end(model: torch.nn.Module):
#     """Mount on_step_end to the model.

#     :param model:The torch module to be pruned
#     :return: hook handle
#     """
#     def hook(module, grad_in, grad_out):
#         for pruning in module.prunings:
#             pruning.on_step_end()
#     hook_handle = model.register_backward_hook(hook)
#     return hook_handle


def _rewrite_optimizer_step(opt):
    """Mount on_before/after_optimizer_step to optimizer.

    :param opt: user optimizer: should be a torch.optim.Optimizer object
    :return: the modified optimizer
    """

    def new_step(self, closure=None):
        if hasattr(self, "prunings"):  ## in case user save the whole optimizer
            for pruning in self.prunings:
                pruning.on_before_optimizer_step()

        if closure is not None:
            res = self.orig_step(closure)
        else:
            res = self.orig_step()
        if hasattr(self, "prunings"):
            for pruning in self.prunings:
                pruning.on_after_optimizer_step()
        return res

    if not isinstance(opt, torch.optim.Optimizer):
        logger.error("User optimizer should be a torch.optim.Optimizer object")

    opt.orig_step = opt.step
    import types

    opt.step = types.MethodType(new_step, opt)

    return opt


def save(obj: object, f, pickle_module=None, pickle_protocol=None, _use_new_zipfile_serialization=None):
    """A rewrite function for torch save.

    :param obj:
    :param f:
    :param pickle_module:
    :param pickle_protocol:
    :param _use_new_zipfile_serialization:
    :return:
    """
    params = {}
    if pickle_module is not None:
        params["pickle_module"] = pickle_module
    if pickle_protocol is not None:
        params["pickle_protocol"] = pickle_protocol
    if _use_new_zipfile_serialization is not None:
        params["_use_new_zipfile_serialization"] = _use_new_zipfile_serialization

    if isinstance(obj, torch.nn.Module) and hasattr(obj, "prunings"):
        prunings = obj.prunings
        obj.prunings = None
        delattr(obj, "prunings")
        obj.inc_hook_handle.remove()
        delattr(obj, "inc_hook_handle")
        if len(params) != 0:
            torch.orig_save(obj, f, params)
        else:
            torch.orig_save(obj, f)
        ##recover
        obj.prunings = prunings
        inc_hook_handle = _register_on_step_begin(obj)
        obj.inc_hook_handle = inc_hook_handle
        return

    if isinstance(obj, torch.optim.Optimizer) and hasattr(obj, "orig_step"):
        prunings = obj.prunings
        obj.prunings = None
        delattr(obj, "prunings")
        obj.step = obj.orig_step
        delattr(obj, "orig_step")
        if len(params) != 0:
            torch.orig_save(obj, f, params)
        else:
            torch.orig_save(obj, f)
        ##recover
        _rewrite_optimizer_step(obj)
        obj.prunings = prunings
        return
    if len(params) != 0:
        torch.orig_save(obj, f, params)
    else:
        torch.orig_save(obj, f)


def _prepare_hooks(model, pruning_list, opt=None):
    """Wrapper the model and optimizer to support all the pruning functionality.

    :param model: The user's model, a torch.nn.Module object
    :param opt: The user's optimizer, a torch.optim object
    :return: The modified model and optimizer
    """
    model.prunings = pruning_list
    if opt is not None:
        opt.prunings = pruning_list
        _rewrite_optimizer_step(opt)

    # Register automated hooks
    inc_hook_handle = _register_on_step_begin(model)
    model.inc_hook_handle = inc_hook_handle
    # Rewrite torch save
    torch.orig_save = torch.save
    setattr(torch, "save", save)


# def complete_pruning(model: torch.nn.Module, opt: torch.optim):
#     """UnWrapper the model and optimizer
#     :param model: the modified model
#     :param opt: the modified optimizer
#     :return: the pruned model and the user's optimizer
#     """

#     model.inc_hook_handle.remove()
#     delattr(model, "inc_hook_handle")
#     for pruning in model.prunings:
#         get_sparsity_ratio(pruning.pruners, model)
#     model.prunings = None
#     delattr(model, "prunings")
#     opt.prunings = None
#     delattr(opt, "prunings")
#     opt.step = opt.orig_step
#     delattr(opt, "orig_step")
#     return model, opt


def prepare_pruning(
    model, config, optimizer=None, dataloader=None, loss_func=None, framework="pytorch", device: str = None
):
    """Get registered pruning class, wrapper the model and optimizer to support all the pruning functionality.

    Get a pruning object from PRUNINGS.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruners information.

    Returns:
        A pruning object.

    Raises: AssertionError: Currently only support prunings that have been registered in PRUNINGS.
    """

    # assert framework in FRAMEWORK.keys(), \
    #         f"does not support {framework}, currently only support framework: {FRAMEWORK.keys()}"
    assert framework == "pytorch", (
        f"The Automation API currently only supports the 'pytorch' framework, "
        f"but the framework given is: {framework}"
    )
    pruning_list = []
    pruning_conf = process_config(config)
    if optimizer is not None:
        basic_conf = []
        for pruner_info in pruning_conf:
            if "gpt" in pruner_info["pruning_type"] or "retrain" in pruner_info["pruning_type"]:
                continue
            basic_conf.append(pruner_info)
        pruning_list.append(PRUNINGS["basic_pruning"](basic_conf, model, optimizer))
        _prepare_hooks(model, pruning_list, opt=optimizer)
    if dataloader is not None:
        # The pruning will be done at initialization time, without inserting any hooks.
        sparse_gpt_conf = []
        retrain_free_conf = []
        for pruner_info in pruning_conf:
            if "gpt" in pruner_info["pruning_type"]:
                sparse_gpt_conf.append(pruner_info)
            elif "retrain" in pruner_info["pruning_type"]:
                retrain_free_conf.append(pruner_info)
        if len(sparse_gpt_conf) > 0:
            pruning_list.append(PRUNINGS["sparse_gpt_pruning"](sparse_gpt_conf, model, dataloader, loss_func, device))
        if len(retrain_free_conf) > 0:
            pruning_list.append(PRUNINGS["retrain_free_pruning"](retrain_free_conf, model, dataloader, loss_func))

    assert len(pruning_list) >= 1, "The pruning config is not standardized and cannot be initialized properly."
    if len(pruning_list) > 1:
        logger.info("Note that more than two pruning algorithms are currently used.")
        return pruning_list
    return pruning_list[0]
