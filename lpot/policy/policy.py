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

POLICIES = {}

def policy_registry(cls):
    """The class decorator used to register all PrunePolicy subclasses.

    Args:
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """
    assert cls.__name__.endswith(
        'PrunePolicy'
    ), "The name of subclass of PrunePolicy should end with \'PrunePolicy\' substring."
    if cls.__name__[:-len('PrunePolicy')].lower() in POLICIES:
        raise ValueError('Cannot have two policies with the same name')
    POLICIES[cls.__name__[:-len('PrunePolicy')].lower()] = cls
    return cls

class PrunePolicy:
    def __init__(self, model, local_config, global_config, adaptor):
        """The base clase of Prune policies

        Args:
            model (object):                        The original model (currently torhc.nn.module
                                                   instance).
            local_config (Conf):                   configs specific for this pruning instance
            global_config (Conf):                  global configs which may be overwritten by 
                                                   local_config
            adaptor (Adaptor):                     Correspond adaptor for current framework

        """
        self.model = model
        self.adaptor = adaptor
        self.tensor_dims = [4]

        if local_config.method:
            self.method = local_config.method
        else:
            self.method = "per_tensor"

        if local_config.init_sparsity:
            self.init_sparsity = local_config["init_sparsity"]
        else:
            self.init_sparsity = global_config.pruning["init_sparsity"]
        if local_config.target_sparsity:
            self.target_sparsity = local_config.target_sparsity
        else:
            self.target_sparsity = global_config.pruning.target_sparsity
        self.start_epoch = global_config.pruning["start_epoch"]
        self.end_epoch = global_config.pruning["end_epoch"]
        self.freq = global_config.pruning["frequency"]
        if local_config.weights:
            self.weights = local_config.weights
        else:
            self.weights = self.adaptor.get_all_weight_names(self.model)

        self.is_last_epoch = False
        self.masks = {}

    def on_epoch_begin(self, epoch):
        raise NotImplementedError

    def on_batch_begin(self, batch_id):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError

    def on_batch_end(self):
        raise NotImplementedError

    def update_sparsity(self, epoch):
        """ update sparsity goals according to epoch numbers

        Args:
            epoch (int): the epoch number

        Returns:
            sprsity (float): sparsity target in this epoch
        """
        if self.start_epoch == self.end_epoch:
            return self.init_sparsity
        if epoch < self.start_epoch:
            return 0
        if epoch > self.end_epoch:
            return self.target_sparsity
        return self.init_sparsity + (self.target_sparsity - self.init_sparsity) * (
            (epoch - self.start_epoch) // self.freq) * self.freq / \
            (self.end_epoch - self.start_epoch)
