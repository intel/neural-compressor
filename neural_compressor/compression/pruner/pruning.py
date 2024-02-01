"""Pruning."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import gc
from typing import Optional

from neural_compressor.compression.pruner.pruners import get_pruner
from neural_compressor.compression.pruner.utils import (
    collect_layer_inputs,
    get_layers,
    get_sparsity_ratio,
    logger,
    parse_to_prune,
    torch,
)

PRUNINGS = {}


def register_pruning(name):
    """Class decorator to register a pruning subclass to the registry.

    Decorator function used before a pruner subclass.
    Make sure that the pruning class decorated by this function can be registered in PRUNINGS.

    Args:
        cls (class): The subclass of register.
        name: A string. Define the pruning type.

    Returns:
        cls: The class of register.
    """

    def register(pruning):
        PRUNINGS[name] = pruning
        return pruning

    return register


class BasePruning:
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

    def __init__(self, config, model, opt=None):
        """Initialize."""
        self._model = model
        self.pruners_info = config
        self.pruners = self._generate_pruners()

    def _generate_pruners(self):
        """Obtain Pruner objects."""
        pruners = []
        # model auto slim related
        # assert isinstance(self._model, torch.nn.Module) # mha only for torch
        from .model_slim.pattern_analyzer import SelfMHASearcher

        for info in self.pruners_info:
            if "mha" in info["pattern"]:
                # head pruning
                pa_obj = SelfMHASearcher(self._model)
                modules, _ = pa_obj.search(split_qkv_ffn=False)
                modules = pa_obj.obtain_mha_module(modules)
                modules = pa_obj.from_layer_name_to_object(modules)
                if len(modules) == 0:
                    logger.warning("one pruner hooks no mha modules, please have a check")
                pruners.append(get_pruner(info, modules))
            else:
                # original pruning types, e.g NxM or N:M
                modules = parse_to_prune(info, self._model)
                if modules == {}:
                    logger.warning("one pruner hooks no layers, please have a check")

                pruners.append(get_pruner(info, modules))
                info["modules"] = [key for key in modules.keys()]
                info["len_of_modules"] = len(info["modules"])
                logger.info(info)

        return pruners

    def on_train_begin(self, dataloader=None):
        """Implement at the beginning of training process.

        Before training, ensure that pruners are generated.
        """
        for pruner in self.pruners:
            pruner.on_train_begin(dataloader)

    def on_step_begin(self, local_step=0):
        """Implement at the beginning of every step."""
        for pruner in self.pruners:
            pruner.on_step_begin(local_step)

    def on_step_end(self):
        """Implement at the end of each step."""
        for pruner in self.pruners:
            pruner.on_step_end()

    def on_before_optimizer_step(self):
        """Implement before optimizer.step()."""
        for pruner in self.pruners:
            pruner.on_before_optimizer_step()

    def on_after_optimizer_step(self):
        """Implement after optimizer.step()."""
        for pruner in self.pruners:
            pruner.on_after_optimizer_step()

    def on_epoch_begin(self, epoch):  # pragma: no cover
        """Implement at the beginning of every epoch."""
        for pruner in self.pruners:
            pruner.on_epoch_begin(epoch)

    def on_epoch_end(self):  # pragma: no cover
        """Implement the end of every epoch."""
        for pruner in self.pruners:
            pruner.on_epoch_end()

    def on_train_end(self):
        """Implement the end of training phase."""
        for pruner in self.pruners:
            pruner.on_train_end()
        get_sparsity_ratio(self.pruners, self._model)


@register_pruning("basic_pruning")
class BasicPruning(BasePruning):
    def __init__(self, config, model, opt=None):
        """Initialize."""
        super().__init__(config, model)


@register_pruning("sparse_gpt_pruning")
class SparseGPTPruning(BasePruning):
    """SparseGPT Pruning
    The SparseGPT pruning_class is derived from BasePruning.

    Args:
        config: A config dict object that contains the pruner information.
        model: The model that need to be pruned.
        dataloader: Processed datasets, which is necessary for sparseGPT pruning.
        device: available device of pruning.
    """

    def __init__(self, config, model, dataloader, framework="pytorch", device=None):
        """Initialize."""
        super().__init__(config, model)
        if device is None:
            self.dev = model.device
        elif isinstance(device, str):
            assert "cpu" in device or "cuda" in device, "Only 'cpu' and 'cuda' are supported."
            self.dev = torch.device(device)
        else:
            assert isinstance(device, torch.device), "Only 'str' and 'torch.device' are supported."
            self.dev = device

        self._layers = []
        self._dataloader = dataloader
        if dataloader is not None:
            self._prepare_pruners()

    def _prepare_pruners(self):
        """One-shot post-training pruning."""
        self.model_dev = self._model.device
        self._layers = get_layers(self._model)
        self._do_pruning()
        self._model = self._model.to(self.model_dev)
        # TODO add get_sparsity_ratio() for sparseGPT

    def gather_single_batch_from_dict(self, data_dict, idx):
        single_batch = {}
        for k, v in data_dict.items():
            single_batch[k] = data_dict[k][idx]
        return single_batch

    def _do_pruning(self):
        from tqdm.auto import tqdm

        layers = self._layers
        self._model = self._model.cpu()
        inputs, positional_inputs, other_input_infos = collect_layer_inputs(
            model=self._model, layers=layers, layer_idx=0, layer_inputs=self._dataloader, device=self.dev
        )
        for i in tqdm(range(len(layers))):
            layer = layers[i].to(self.dev)
            layer_index_str = "." + str(i) + "."
            handles_list = []
            for pruner in self.pruners:
                layer_op_names = [key for key in pruner.modules.keys() if layer_index_str in key]
                if bool(layer_op_names):
                    handles_list.append(pruner.register_gpt_hook(layer_op_names))
            prune_flag = bool(handles_list)
            if prune_flag:
                for j in range(len(inputs)):
                    other_infos = self.gather_single_batch_from_dict(other_input_infos, j)
                    with torch.no_grad():
                        layer(inputs[j], *positional_inputs, **other_infos)[0]
                for handles in handles_list:
                    for h in handles:
                        h.remove()
                for pruner in self.pruners:
                    layer_op_names = [key for key in pruner.modules.keys() if layer_index_str in key]
                    pruner.fasterprune(layer_op_names)
            for j in range(len(inputs)):
                # the weights of current layer have been pruned, get the latest outputs as the inputs for next layer
                other_infos = self.gather_single_batch_from_dict(other_input_infos, j)
                with torch.no_grad():
                    inputs[j] = layer(inputs[j], *positional_inputs, **other_infos)[0]
            layers[i] = layer.cpu()
            if "cuda" in self.dev.type:
                torch.cuda.empty_cache()
        del other_infos
        del positional_inputs
        del inputs
        gc.collect()
        if "cuda" in self.dev.type:
            torch.cuda.empty_cache()

    def on_train_begin(self, dataloader):  # pragma: no cover
        if self._dataloader is not None:
            logger.info(
                "The sparseGPT pruning is already done at initialization time, "
                "calling on_train_begin() is a redundant operation."
            )
        elif dataloader is None:
            logger.error(
                "The sparseGPT pruning must be passed the 'dataloader' argument "
                "when initializing or calling on_train_begin()"
            )
        self._dataloader = dataloader
        self._prepare_pruners()


@register_pruning("retrain_free_pruning")
class RetrainFreePruning(BasePruning):
    def __init__(self, config, model, dataloader=None, loss_func=None, framework="pytorch"):
        """Initialize."""
        super().__init__(config, model)
        self._dataloader = dataloader
        self._loss_func = loss_func
        if dataloader is not None:
            self._prepare_pruners()

    def _prepare_pruners(self):
        self._do_pruning()
        get_sparsity_ratio(self.pruners, self._model)

    def _do_pruning(self):
        from tqdm.auto import tqdm

        length = len(self._dataloader.dataset)
        if self._dataloader.batch_sampler is not None:
            length = len(self._dataloader.batch_sampler)
        progress_bar = tqdm(range(length))
        if self._loss_func is not None:
            for inputs, target in self._dataloader:
                self.on_step_begin()
                outputs = self._model(inputs)
                loss = self._loss_func(outputs, target)
                loss.backward()
                self.on_step_end()
                progress_bar.update(1)
        else:
            for batch in self._dataloader:
                self.on_step_begin()
                outputs = self._model(return_dict=True, **batch)
                loss = outputs.loss
                loss.backward()
                self.on_step_end()
                progress_bar.update(1)

    # def on_train_begin(self, dataloader):
    #     if self._dataloader is not None:
    #         logger.info("The retrain_free pruning is already done at initialization time, " \
    #                     "calling on_train_begin() is a redundant operation.")
    #     elif dataloader is None:
    #         logger.error("The retrain_free pruning must be passed the 'dataloader' argument " \
    #                      "when initializing or calling on_train_begin()")
    #     self._dataloader = dataloader
    #     self._prepare_pruners()
