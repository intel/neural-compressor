#!/usr/bin/env python
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

import torch.nn

from .prune_utils import process_config, parse_to_prune, parse_not_to_prune
from .pruner import get_pruner
from .logger import logger

class Pruning:
    def __init__(self, config):
        self.model = None
        self.config_file_path = config
        self.pruners = []
        self.pruner_info = process_config(self.config_file_path)

    def update_items_for_all_pruners(self, **kwargs):
        for item in self.pruner_info:
            for key in kwargs:
                if key in item.keys():
                    item[key] = kwargs[key]

    #def _call_pruners(self, func):
    #    def warpper(self, *args, **kw):
    #        func_name = f"{func.__name__}"
    #        func(self, *args, **kw)
    #        for prune in self.pruners:
    #            prun_func = getattr(prune, func_name)
    #            prun_func(*args, **kw)
    #
    #    return warpper

    def _generate_pruners(self):
        assert isinstance(self.model, torch.nn.Module)

        for info in self.pruner_info:
            modules = parse_to_prune(self.model, info)
            modules = parse_not_to_prune(modules, info)
            if modules == {}:
                logger.warning("one pruner hooks no layers, please have a check")

            self.pruners.append(get_pruner(modules, info))
            info['modules'] = [key for key in modules.keys()]
            info['len_of_modules'] = len(info['modules'])
            logger.info(info)

    #@_call_pruners
    def on_train_begin(self):
        self._generate_pruners()  ##TODO is there better place to place

    #@_call_pruners
    def on_epoch_begin(self, epoch):
        for pruner in self.pruners:
            pruner.on_epoch_begin(epoch)      

    #@_call_pruners
    def on_step_begin(self, local_step):
        for pruner in self.pruners:
            pruner.on_step_begin(local_step)

    #@_call_pruners
    def on_before_optimizer_step(self):
        for pruner in self.pruners:
            pruner.on_before_optimizer_step()

    #@_call_pruners
    def on_step_end(self):
        for pruner in self.pruners:
            pruner.on_step_end()

    #@_call_pruners
    def on_epoch_end(self):
        for pruner in self.pruners:
            pruner.on_epoch_end()

    #@_call_pruners
    def on_train_end(self):
        for pruner in self.pruners:
            pruner.on_train_end()

    #@_call_pruners
    def on_after_optimizer_step(self):
        for pruner in self.pruners:
            pruner.on_after_optimizer_step()
