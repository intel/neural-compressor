#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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
from .policy import policy_registry, PrunePolicy
from ..utils import logger


@policy_registry
class MagnitudePrunePolicy(PrunePolicy):
    def __init__(self, model, local_config, global_config, adaptor):
        super(MagnitudePrunePolicy, self).__init__(model, local_config, global_config, adaptor)

    def on_epoch_begin(self, epoch):
        logger.debug("start pruning in epoch {}".format(str(epoch)))
        self.sparsity = self.update_sparsity(epoch)
        self.is_last_epoch = epoch == self.end_epoch
        logger.debug("epoch {} sparsity = {}".format(str(epoch), str(self.sparsity)))
        if epoch >= self.start_epoch and epoch <= self.end_epoch:
            self.compute_mask()

    def on_batch_begin(self, batch_id):
        for weight in self.weights:
            if weight in self.masks:
                new_weight = self.masks[weight] * \
                    np.array(self.adaptor.get_weight(self.model, weight))
                new_weight_zeros = (new_weight == 0).sum()
                self.adaptor.update_weights(self.model, weight, new_weight)

    def compute_mask(self):
        """compute masks according to absolute values"""
        for weight in self.weights:
            tensor = np.array(self.adaptor.get_weight(self.model, weight))
            if len(tensor.shape) in self.tensor_dims:
                if self.method == "per_channel":
                    tensor_flat = tensor.copy().reshape([tensor.shape[0], tensor.shape[1], -1])
                    tensor_flat.sort(axis=-1)
                    threshold = tensor_flat[:, :, int(self.sparsity * tensor_flat.shape[-1])]
                    threshold = np.expand_dims(np.expand_dims(threshold, -1), -1)
                    threshold = np.repeat(threshold, tensor.shape[-1], axis=-1)
                    threshold = np.repeat(threshold, tensor.shape[-2], axis=-2)
                    self.masks[weight] = threshold < tensor
                else:
                    tensor_flat = sorted(np.abs(tensor.flatten()))
                    threshold = float(tensor_flat[int(tensor_flat.size * self.sparsity)])
                    self.masks[weight] = threshold < np.abs(tensor)

    def on_epoch_end(self):
        if self.is_last_epoch:
            for weight in self.weights:
                if weight in self.masks:
                    logger.info(
                        "{} with mask sparsity {} {} {}".format(
                            weight, str(
                            self.masks[weight].size), str(
                            self.masks[weight].sum()), str(
                            1 - self.masks[weight].sum() / self.masks[weight].size)))
                    new_weight = self.masks[weight] * \
                        np.array(self.adaptor.get_weight(self.model, weight))
                    self.adaptor.update_weights(self.model, weight, new_weight)

    def on_batch_end(self):
        for weight in self.weights:
            if weight in self.masks:
                new_weight = self.masks[weight] * \
                    np.array(self.adaptor.get_weight(self.model, weight))
                self.adaptor.update_weights(self.model, weight, new_weight)
