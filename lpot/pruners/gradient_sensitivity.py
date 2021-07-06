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

import numpy as np
from .pruner import pruner_registry, Pruner
from heapq import heappush, heappop
from ..utils import logger
import re

@pruner_registry
class GradientSensitivityPruner(Pruner):
    def __init__(self, model, local_config, global_config):
        super(GradientSensitivityPruner, self).__init__(model, local_config, global_config)
        self.parameters = local_config.parameters
        self.importance = {}

    def on_epoch_begin(self, epoch):
        pass

    def on_batch_begin(self, batch_id):
        pass

    def on_epoch_end(self):
        for weight_name_raw in self.weights:
            for weight_name in self.parse_weight_name(weight_name_raw):
                self.prune_weight(self.model,
                                  self.importance,
                                  weight_name,
                                  self.parameters)

    def parse_weight_name(self, weight_name_pattern):
        # check if asterisk is used to match bert layer indexes
        if '*' not in weight_name_pattern:
            yield weight_name_pattern
        else:
            weight_all_names = self.model.get_all_weight_names()
            importance_inputs = self.parameters['importance_inputs']
            for single_weight_name in weight_all_names:
                index_group = re.match(
                        weight_name_pattern.replace('*', '(\d+)'), single_weight_name)
                if index_group is not None:
                    index = index_group.group(1)
                    if self.parameters.get(index) is None:
                        self.parameters['index'] = int(index)
                    # dynamic change importance_inputs with matched index
                    self.parameters['importance_inputs'] = [
                            x.replace('*', index) for x in self.parameters['importance_inputs']]
                    yield single_weight_name
                    # change importance_inputs back
                    self.parameters['importance_inputs'] = importance_inputs

    def on_batch_end(self):
        for weight_name_raw in self.weights:
            for weight_name in self.parse_weight_name(weight_name_raw):
                if self.parameters['importance_metric'] == 'abs_gradient':
                    self.update_importance_abs(self.model,
                                               self.importance,
                                               weight_name,
                                               self.parameters)
                elif self.parameters['importance_metric'] == 'weighted_gradient':
                    self.update_importance_weighted(self.model,
                                                    self.importance,
                                                    weight_name,
                                                    self.parameters)

    def prune_weight(self, model, importance, weight_name, parameters):
        if parameters['normalize']:
            exponent = 2
            norm_by_layer = np.power(
                    np.power(importance[weight_name], exponent).sum(-1), 1 / exponent)
            importance[weight_name] /= np.expand_dims(norm_by_layer, -1) + 1e-20
        importance = importance[weight_name]
        weight_tensor = np.array(model.get_weight(weight_name))
        if parameters['transpose']:
            weight_tensor = weight_tensor.transpose((1, 0))

        weight_tensor = self.prune_by_importance(weight_tensor,
                                                 importance,
                                                 parameters['target'],
                                                 parameters['stride'])
        if parameters['transpose']:
            weight_tensor = weight_tensor.transpose((1, 0))

        model.update_weights(weight_name, weight_tensor)

    def update_importance_abs(self, model, importance, weight_name, parameters):
        head_mask = model.get_inputs(
                input_name=parameters['importance_inputs'][0])
        if importance.get(weight_name) is not None:
           importance[weight_name] += np.absolute(
                np.array(model.get_gradient(head_mask)))[parameters['index']]
        else:
            importance[weight_name] = np.absolute(
                np.array(model.get_gradient(head_mask)))[parameters['index']]

    def update_importance_weighted(self, model, importance, weight_name, parameters):
        def weighted_grad(input_weight):
            weight_grad = np.array(model.get_gradient(input_weight))
            weight = np.array(model.get_weight(input_weight))
            weighted_grad = weight_grad * weight
            # TODO: add more check here
            if weighted_grad.ndim > 1:
                weighted_grad = weighted_grad.sum(1)
            return weighted_grad

        accumulated_grad = sum([weighted_grad(input_weight) for input_weight \
                            in parameters['importance_inputs']])

        if importance.get(weight_name) is not None:
            importance[weight_name] += np.absolute(accumulated_grad)
        else:
            importance[weight_name] = np.absolute(accumulated_grad)

    def prune_by_importance(self, tensor, importance, num_instances, stride):
        # structured prune
        importance_ordered = []
        i = 0
        for heads in importance:
            heappush(importance_ordered, (-heads, i))
            i += 1
        sorted_tensor_to_concat = None
        i = 0
        while importance_ordered and i < num_instances:
            head_to_add = heappop(importance_ordered)[1]
            if sorted_tensor_to_concat is None:
                sorted_tensor_to_concat = (
                        tensor[int(head_to_add * stride): int(head_to_add * stride) +
                               int(stride), ...], )
            else:
                sorted_tensor_to_concat += (
                        tensor[int(head_to_add * stride): int(head_to_add * stride) +
                               int(stride), ...], )
            i += 1
        return np.concatenate(sorted_tensor_to_concat)
