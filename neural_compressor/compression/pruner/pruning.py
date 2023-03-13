"""Pruning."""
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

from .utils import process_config, parse_to_prune, \
    check_config, update_params
from .pruners import get_pruner
from .utils import logger, get_sparsity_ratio, torch
import re
import torch
from typing import Iterable, Union, Callable, Optional, List


def _generate_pruners(config, model):
    """Obtain Pruner objects."""
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


def register_on_step_begin(model):
    def hook(module, input):
        for pruner in module.pruners:
            pruner.on_step_begin(0)

    hook_handle = model.register_forward_pre_hook(hook)
    return hook_handle


class PruningOptimizer(torch.optim.Optimizer):
    def __init__(self, orig_opt):
        self.inc_opt = orig_opt

    def step(self, closure: Optional[Callable[[], float]] = ...):
        for pruner in self.inc_opt.pruners:
            pruner.on_before_optimizer_step()
        res = self.orig_step(closure)
        for pruner in self.inc_opt.pruners:
            pruner.on_after_optimizer_step()
        return res


def rewrite_optimizer_step(opt: torch.optim.Optimizer):
    opt = PruningOptimizer(opt)
    return opt


def PruningWrapper(config, model: torch.nn.Module, opt: torch.optim):
    pruners = _generate_pruners(config, model)  ##list
    model.pruners = pruners
    opt.pruners = pruners

    inc_hook_handle = register_on_step_begin(model)
    model.inc_hook_handle = inc_hook_handle
    rewrite_optimizer_step(opt)
    return model, opt


def PurningUnWrapper( model: torch.nn.Module, opt: torch.optim):
    model.hook_handel.remove()
    delattr(model,"inc_hook_handle")
    return model, opt.inc_opt

