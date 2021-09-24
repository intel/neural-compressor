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

from .pruner import pruner_registry, Pruner


@pruner_registry
class PatternLockPruner(Pruner):
    def __init__(self, model, local_config, global_config):
        super(PatternLockPruner, self).__init__(model, local_config, global_config)
        self.compute_mask()

    def on_epoch_begin(self, epoch):
        pass

    def on_batch_begin(self, batch_id):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_end(self):
        self.update_weights()

    def compute_mask(self):
        """compute masks according to current sparsity pattern"""
        for weight in self.weights:
            tensor = self.model.get_weight(weight)
            if len(tensor.shape) in self.tensor_dims:
                self.masks[weight] = tensor == 0.

    def update_weights(self):
        for weight in self.weights:
            if weight in self.masks:
                self.model.prune_weights_(weight, self.masks[weight])
