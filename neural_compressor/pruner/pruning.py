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
from neural_compressor.utils.utility import LazyImport

LazyImport('torch.nn')
torch = LazyImport('torch')

from neural_compressor.pruner.utils import process_config, parse_to_prune, \
    check_config, update_params
from neural_compressor.pruner.pruners import get_pruner
from neural_compressor.utils import logger
import re
from neural_compressor.config import WeightPruningConfig


class Pruning:
    """Pruning.

    The main class to do pruning; it contains at least one Pruner object.

    Args:
        config: a string representing the path to a config file. For config file template, please refer to
            https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager/

    Attributes:
        model: The model object to prune.
        config_file_path: A string representing the path to a config file.
        pruners: A list. A list of Pruner objects.
        pruner_info: A config dict object that contains pruners' information.
    """

    def __init__(self, config):
        """Initialize."""
        self._model = None
        self.pruners = []
        self.pruners_info = process_config(config)

    @property
    def model(self):
        """Obtain model in neural_compressor.model."""
        return self._model

    @model.setter
    def model(self, user_model):
        self._model = user_model

    def update_config(self, *args, **kwargs):
        """Add user-defined arguments to the original configurations.

        The original config of pruning is read from a file.
        However, users can still modify configurations by passing key-value arguments in this function.
        Please note that the key-value arguments' keys could be processed in current configuration.
        """
        for item in self.pruners_info:
            for key in kwargs:
                if key in item.keys():
                    item[key] = kwargs[key]

            update_params(item)
            check_config(item)

    def get_sparsity_ratio(self):
        """Calculate sparsity ratio of a module/layer.

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
        for name, module in self._model.named_modules():
            if type(module).__name__ in ["Linear"] or re.search(r'Conv.d', type(module).__name__) != None:
                linear_conv_cnt += module.weight.numel()

        for n, param in self._model.named_parameters():
            param_cnt += param.numel()
        if linear_conv_cnt == 0:
            blockwise_over_matmul_gemm_conv = 0
            elementwise_over_matmul_gemm_conv = 0
        else:
            blockwise_over_matmul_gemm_conv = float(pattern_sparsity_cnt) / linear_conv_cnt
            elementwise_over_matmul_gemm_conv = float(element_sparsity_cnt) / linear_conv_cnt
        if param_cnt == 0:
            elementwise_over_all = 0
        else:
            elementwise_over_all = float(
                element_sparsity_cnt) / param_cnt

        return elementwise_over_matmul_gemm_conv, elementwise_over_all, blockwise_over_matmul_gemm_conv

    def _generate_pruners(self):
        """Obtain Pruner objects."""
        assert isinstance(self._model, torch.nn.Module)

        for info in self.pruners_info:
            modules = parse_to_prune(info, self._model)
            if modules == {}:
                logger.warning("one pruner hooks no layers, please have a check")

            self.pruners.append(get_pruner(info, modules))
            info['modules'] = [key for key in modules.keys()]
            info['len_of_modules'] = len(info['modules'])
            logger.info(info)

    def on_train_begin(self):
        """Implement at the beginning of training process.

        Before training, ensure that pruners are generated.
        """
        self._generate_pruners()  ##TODO is there better place to place

    def on_epoch_begin(self, epoch):
        """Implement at the beginning of every epoch."""
        for pruner in self.pruners:
            pruner.on_epoch_begin(epoch)

    def on_step_begin(self, local_step):
        """Implement at the beginning of every step."""
        for pruner in self.pruners:
            pruner.on_step_begin(local_step)

    def on_before_optimizer_step(self):
        """Implement before optimizer.step()."""
        for pruner in self.pruners:
            pruner.on_before_optimizer_step()

    def on_step_end(self):
        """Implement at the end of every step."""
        for pruner in self.pruners:
            pruner.on_step_end()

    def on_epoch_end(self):
        """Implement the end of every epoch."""
        for pruner in self.pruners:
            pruner.on_epoch_end()

    def on_train_end(self):
        """Implement the end of training phase."""
        for pruner in self.pruners:
            pruner.on_train_end()
        self.get_sparsity_ratio()

    def on_before_eval(self):
        """Implement at the beginning of evaluation phase."""
        for pruner in self.pruners:
            pruner.on_before_eval()

    def on_after_eval(self):
        """Implement at the end of evaluation phase."""
        for pruner in self.pruners:
            pruner.on_after_eval()

    def on_after_optimizer_step(self):
        """Implement after optimizer.step()."""
        for pruner in self.pruners:
            pruner.on_after_optimizer_step()
