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

from ..experimental.pruning_recipes.patterns import patterns

PRUNERS = {}

def pruner_registry(cls):
    """The class decorator used to register all Pruners subclasses.

    Args:
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """
    assert cls.__name__.endswith(
        'Pruner'
    ), "The name of subclass of Pruner should end with \'Pruner\' substring."
    if cls.__name__[:-len('Pruner')].lower() in PRUNERS:
        raise ValueError('Cannot have two pruner with the same name')
    PRUNERS[cls.__name__[:-len('Pruner')]] = cls
    return cls

class Pruner:
    def __init__(self, model, local_config, global_config):
        """The base clase of Pruner

        Args:
            model (object):          The original model (currently PyTorchModel instance).
            local_config (Conf):     configs specific for this pruning instance
            global_config (Conf):    global configs which may be overwritten by
                                     local_config

        """
        self.model = model
        #2 for linear weight, 4 for conv weight
        self.tensor_dims = [2, 4]

        if local_config.method:
            self.method = local_config.method
        else:
            self.method = "per_tensor"

        if local_config.initial_sparsity:
            self.initial_sparsity = local_config.initial_sparsity
        else:
            self.initial_sparsity = global_config.initial_sparsity
        if local_config.target_sparsity:
            self.target_sparsity = local_config.target_sparsity
        else:
            self.target_sparsity = global_config.target_sparsity
        if local_config.start_epoch:
            self.start_epoch = local_config.start_epoch
        else:
            self.start_epoch = global_config.start_epoch
        if local_config.end_epoch:
            self.end_epoch = local_config.end_epoch
        else:
            self.end_epoch = global_config.end_epoch
        if local_config.update_frequency:
            self.freq = local_config.update_frequency
        else:
            self.freq = global_config.update_frequency
        if local_config.names:
            self.weights = local_config.names
        else:
            self.weights = self.model.get_all_weight_names()

        self.is_last_epoch = False

        # TBD, add pattern in config
        if hasattr(local_config, 'pattern'):
            self.pattern = patterns[local_config.pattern]()
        else:
            self.pattern = patterns['tile_pattern_1x1']()
        self.masks = {}

    def on_epoch_begin(self, epoch):
        raise NotImplementedError

    def on_batch_begin(self, batch_id):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError

    def on_batch_end(self):
        raise NotImplementedError

    def on_post_grad(self):
        pass

    def update_sparsity(self, epoch):
        """ update sparsity goals according to epoch numbers

        Args:
            epoch (int): the epoch number

        Returns:
            sparsity (float): sparsity target in this epoch
        """
        if self.start_epoch == self.end_epoch:
            return self.initial_sparsity
        if epoch < self.start_epoch:
            return 0
        if epoch > self.end_epoch:
            return self.target_sparsity
        return self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * (
            (epoch - self.start_epoch) // self.freq) * self.freq / \
            (self.end_epoch - self.start_epoch)
