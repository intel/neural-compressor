"""Regularizer."""

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

from .patterns.base import PytorchBasePattern
from .utils import torch

REGS = {}


def register_reg(name):
    """Register a regularizator to the registry.

    Args:
        name: A string that defines the scheduler type.

    Returns:
        cls: The class of register.
    """

    def register(reg):
        REGS[name] = reg
        return reg

    return register


def get_reg_type(config):
    """Obtain the regularizer type.

    Args:
        config: A config dict object that includes information of the regularizer.
    """
    for key in REGS.keys():  ##assume there is only one reg
        if config.get(key, None) is not None:
            return key
    return None


def get_reg(config, modules, pattern):
    """Get registered regularizator class.

    Args:
        config: A config dict object that includes information of the regularizer.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        pattern: A config dict object that includes information of the pattern.
    """
    reg_type = config["reg_type"]
    if reg_type is None:
        return BaseReg(config, modules, pattern)
    if reg_type not in REGS.keys():
        assert False, f"regularizator does not support {reg_type}, currently only support {REGS.keys()}"
    return REGS[reg_type](config, modules, pattern, config["reg_coeff"])


class BaseReg:
    """Regularizer.

    The class that performs regularization.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that includes information of the regularizer.
        pattern: A config dict object that includes information of the pattern.
    """

    def __init__(self, config: dict, modules: dict, pattern: PytorchBasePattern):
        """Initialize."""
        self.modules = modules
        self.config = config
        self.pattern = pattern

    def on_before_optimizer_step(self):
        """Implement before optimizer.step()."""
        pass

    def on_after_optimizer_step(self):
        """Implement after optimizer.step()."""
        pass


@register_reg("group_lasso")
class GroupLasso(BaseReg):
    """Regularizer.

    A regularizer class derived from BaseReg. In this class, the Group-lasso regularization will be performed.
    Group-lasso is a variable-selection and regularization method.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that includes information of the regularizer.
        pattern: A config dict object that includes information of the pattern.

    Attributes:
        reg_terms: A dict {"module_name": Tensor} of regularization terms.
        alpha: A float representing the coefficient related to group lasso.
    """

    def __init__(self, config: dict, modules: dict, pattern: PytorchBasePattern, coeff):
        """Initialize."""
        super(GroupLasso, self).__init__(config, modules, pattern)
        assert "x" in self.config.pattern, "group lasso only supports NXM pattern"
        self.reg_terms = {}
        self.alpha = float(coeff)
        assert self.alpha >= 0, "group lasso only supports positive coeff"

    def on_before_optimizer_step(self):
        """Calculate the group-lasso score map."""
        with torch.no_grad():
            if self.pattern.invalid_layers is None:
                self.pattern.check_layer_validity()
            for key in self.modules.keys():
                if key in self.pattern.invalid_layers:
                    continue
                grad = self.modules[key].weight.grad
                reg_term = self.pattern.reshape_orig_to_pattern(grad, key)
                reg_term = self.alpha / (torch.norm(reg_term, p=2, dim=[1, 3]) + 1e-12)
                reg_term[torch.isinf(reg_term)] = 0.0
                self.reg_terms[key] = reg_term

    def on_after_optimizer_step(self):  ##decoupled with grad descent
        """Perform group lasso regularization after optimization."""
        with torch.no_grad():
            for key in self.modules.keys():
                if key in self.pattern.invalid_layers:
                    continue
                reg_term = self.pattern.reshape_reduced_to_orig(
                    self.reg_terms[key], key, self.modules[key].weight.shape
                )
                self.modules[key].weight -= reg_term * self.modules[key].weight
