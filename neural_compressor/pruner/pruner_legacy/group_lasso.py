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

"""Group Lasso pruner."""

import re
import copy
import numpy as np
from .pruner import pruner_registry, Pruner
from .magnitude import BasicMagnitudePruner
from neural_compressor.utils import logger

@pruner_registry
class GroupLassoPruner(BasicMagnitudePruner):
    """Group Lasso pruner class.

    Args:
        model (object): The original model (currently PyTorchModel instance).
        local_config (Conf): configs specific for this pruning instance.
        global_config (Conf): global configs which may be overwritten by local_config.
    """

    def __init__(self, model, local_config, global_config):
        """Initialize the attributes."""
        super(GroupLassoPruner, self).__init__(model, local_config, global_config)
        self.cur_weights = copy.deepcopy(self.weights)
        self.is_masks_set = False
        self.alpha = local_config.parameters['alpha']

    def on_before_optimizer_step(self):
        """Update gradient to prune the weights by back propagation."""
        if self.cur_weights:
            for weight_name in self.weights:
                weight_grad = self.model.get_gradient(weight_name)
                weight = np.array(self.model.get_weight(weight_name))
                reshaped_weight = self.pattern.reshape(weight)
                coeff = self.alpha / np.linalg.norm(reshaped_weight, 2, axis=(1,3))
                coeff[np.isinf(coeff)] = 0
                coeff = self.pattern.repeat_mask(coeff).reshape(weight.shape)
                weight_grad += coeff * weight
                self.model.update_gradient(weight_name, weight_grad)

            for weight_name in self.weights:
                weight = self.model.get_weight(weight_name)
                grad = self.model.get_gradient(weight_name)
                grad[weight == 0] = 0
                self.model.update_gradient(weight_name, grad)
        else:
            for weight_name in self.weights:
                grad = self.model.get_gradient(weight_name)
                new_grad = grad * self.masks[weight_name]
                self.model.update_gradient(weight_name, new_grad)
