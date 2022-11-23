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
    """Pruning.
    
    The main class that users will used in codes to do pruning.
    Contain at least one Pruner object.
    
    Args:
        config: a string. The path to a config file. For config file template, please refer to
            https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager/
    
    Attributes:
        model: The model object to prune.
        config_file_path: A string. The path to a config file.
        pruners: A list. A list of Pruner objects.
        pruner_info: A config dict object. Contains pruners' information.    
    """
    
    def __init__(self, config):
        self.model = None
        self.config_file_path = config
        self.pruners = []
        self.pruner_info = process_config(self.config_file_path)

    def update_items_for_all_pruners(self, **kwargs):
        """Functions which add User-defined arguments to the original configurations.
        
        The original config of pruning is read from a file. 
        However, users can still modify configurations by passing key-value arguments in this function.
        Please note that the key-value arguments' keys are analysable in current configuration.
        """
        for item in self.pruner_info:
            for key in kwargs:
                if key in item.keys():
                    item[key] = kwargs[key]

    # def _call_pruners(self, func):
    #     """Function which decorates the Pruning class's functions.
    #     
    #     It can simplify codes by calling same-name functions in Pruning's Pruner objects.
    #     For example, when it decorates on_step_begin function of Pruning, 
    #         it automatically calls its Pruners' on_step_begin functions without a "for" code.
    #     However, when this trick is enabled, the pylint validation on INC cannot passed, therefore commented out.
    #     """
    #    def warpper(self, *args, **kw):
    #        func_name = f"{func.__name__}"
    #        func(self, *args, **kw)
    #        for prune in self.pruners:
    #            prun_func = getattr(prune, func_name)
    #            prun_func(*args, **kw)
    #
    #    return warpper

    def get_sparsity_ratio(self):
        """Functions that calculate a modules/layers sparsity.
        
        Returns:
            Three floats.
            elementwise_over_matmul_gemm_conv refers to zero elements' ratio in pruning layers.
            elementwise_over_all refers to zero elements' ratio in all layers in the model.
            blockwise_over_matmul_gemm_conv refers to all-zero blocks' ratio in pruning layers.
        """
        pattern_sparsity_cnt = 0
        element_sparsity_cnt = 0
        for pruner in self.pruners:
            modules = pruner.modules
            sparsity_ratio = pruner.pattern.get_sparsity_ratio(pruner.masks)
            cnt = 0
            for key in modules.keys():
                cnt += modules[key].weight.numel()
            pattern_sparsity_cnt += int(cnt * sparsity_ratio)
            for key in pruner.masks.keys():
                element_sparsity_cnt += torch.sum(pruner.masks[key] == 0).data.item()

        linear_conv_cnt = 0
        param_cnt = 0
        for name, module in self.model.named_modules():
            if type(module).__name__ in ["Linear"] or "Conv" in type(module).__name__:
                linear_conv_cnt += module.weight.numel()

        for n, param in self.model.named_parameters():
            param_cnt += param.numel()
        blockwise_over_matmul_gemm_conv = float(pattern_sparsity_cnt) / linear_conv_cnt
        elementwise_over_matmul_gemm_conv = float(element_sparsity_cnt) / linear_conv_cnt
        elementwise_over_all = float(
            element_sparsity_cnt) / param_cnt

        return elementwise_over_matmul_gemm_conv, elementwise_over_all, blockwise_over_matmul_gemm_conv

    def _generate_pruners(self):
        assert isinstance(self.model, torch.nn.Module)

        for info in self.pruner_info:
            modules = parse_to_prune(info, self.model)
            modules = parse_not_to_prune(info, modules)
            if modules == {}:
                logger.warning("one pruner hooks no layers, please have a check")

            self.pruners.append(get_pruner(info, modules))
            info['modules'] = [key for key in modules.keys()]
            info['len_of_modules'] = len(info['modules'])
            logger.info(info)

    # @_call_pruners
    def on_train_begin(self):
        self._generate_pruners()  ##TODO is there better place to place

    # @_call_pruners
    def on_epoch_begin(self, epoch):
        for pruner in self.pruners:
            pruner.on_epoch_begin(epoch)


    # @_call_pruners
    def on_step_begin(self, local_step):
        for pruner in self.pruners:
            pruner.on_step_begin(local_step)

    # @_call_pruners
    def on_before_optimizer_step(self):
        for pruner in self.pruners:
            pruner.on_before_optimizer_step()

    # @_call_pruners
    def on_step_end(self):
        for pruner in self.pruners:
            pruner.on_step_end()

    # @_call_pruners
    def on_epoch_end(self):
        for pruner in self.pruners:
            pruner.on_epoch_end()

    # @_call_pruners
    def on_train_end(self):
        for pruner in self.pruners:
            pruner.on_train_end()

        # @_call_pruners

    def on_before_eval(self):
        for pruner in self.pruners:
            pruner.on_before_eval()

    def on_after_eval(self):
        for pruner in self.pruners:
            pruner.on_after_eval()

    # @_call_pruners
    def on_after_optimizer_step(self):
        for pruner in self.pruners:
            pruner.on_after_optimizer_step()
