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

from neural_compressor.compression.pruner.utils import process_config, parse_to_prune, get_sparsity_ratio
from neural_compressor.compression.pruner.pruners import get_pruner
from neural_compressor.compression.pruner.utils import logger, torch, collect_layer_inputs, get_layers
from typing import Optional
from tqdm.auto import tqdm
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
    
    def __init__(self, config, model: torch.nn.Module):
        """Initialize."""
        self._model = model
        self.pruners_info = config
        self.pruners = self._generate_pruners(model)

    def _generate_pruners(self, model: torch.nn.Module):
        """Generate pruners.

        :param config: WeightPruningConfig
        :param model: The torch module to be pruned
        :return: A list of pruner
        """
        assert isinstance(model, torch.nn.Module)
        pruners = []
        for info in self.pruners_info:
            modules = parse_to_prune(info, model)
            if modules == {}:
                logger.warning("one pruner hooks no layers, please have a check")
            pruners.append(get_pruner(info, modules))
            info['modules'] = [key for key in modules.keys()]
            info['len_of_modules'] = len(info['modules'])
            logger.info(info)
        return pruners


@register_pruning("basic_pruning")
class BasicPruning(BasePruning):
    def __init__(self, config, model: torch.nn.Module, opt: torch.optim.Optimizer):
        """Initialize."""
        super().__init__(config, model)
        self._prepare_pruners(model, opt)
    
    def _register_on_step_begin(self, model: torch.nn.Module):
        """Mount on_step_begin to the model.

        :param model:The torch module to be pruned
        :return: hook handle
        """

        def hook(module, input):
            for pruner in module.pruners:
                pruner.on_step_begin(0)

        hook_handle = model.register_forward_pre_hook(hook)
        return hook_handle

    def _rewrite_optimizer_step(self, opt: torch.optim.Optimizer):
        """Mount on_before/after_optimizer_step to optimizer.

        :param opt: user optimizer: should be a torch.optim.Optimizer object
        :return: the modified optimizer
        """

        def new_step(self, closure=None):
            if hasattr(self, "pruners"):  ## in case user save the whole optimzer
                for pruner in self.pruners:
                    pruner.on_before_optimizer_step()

            if closure is not None:
                res = self.orig_step(closure)
            else:
                res = self.orig_step()
            if hasattr(self, "pruners"):
                for pruner in self.pruners:
                    pruner.on_after_optimizer_step()
            return res

        opt.orig_step = opt.step
        import types
        opt.step = types.MethodType(new_step, opt)
        return opt

    def save(self,
            obj: object,
            f,
            pickle_module=None,
            pickle_protocol=None,
            _use_new_zipfile_serialization=None
    ):
        """A rewrite function for torch save.

        :param obj:
        :param f:
        :param pickle_module:
        :param pickle_protocol:
        :param _use_new_zipfile_serialization:
        :return:
        """
        params = {}
        if pickle_module != None:
            params['pickle_module'] = pickle_module
        if pickle_protocol != None:
            params['pickle_protocol'] = pickle_protocol
        if _use_new_zipfile_serialization != None:
            params['_use_new_zipfile_serialization'] = _use_new_zipfile_serialization

        if isinstance(obj, torch.nn.Module) and hasattr(obj, "pruners"):
            pruners = obj.pruners
            obj.pruners = None
            delattr(obj, "pruners")
            obj.inc_hook_handle.remove()
            delattr(obj, "inc_hook_handle")
            if len(params) != 0:
                torch.orig_save(obj, f, params)
            else:
                torch.orig_save(obj, f)
            ##recover
            obj.pruners = pruners
            inc_hook_handle = self._register_on_step_begin(obj)
            obj.inc_hook_handle = inc_hook_handle
            return

        if isinstance(obj, torch.optim.Optimizer) and hasattr(obj, "orig_step"):
            pruners = obj.pruners
            obj.pruners = None
            delattr(obj, "pruners")
            obj.step = obj.orig_step
            delattr(obj, "orig_step")
            if len(params) != 0:
                torch.orig_save(obj, f, params)
            else:
                torch.orig_save(obj, f)
            ##recover
            self._rewrite_optimizer_step(obj)
            obj.pruners = pruners
            return
        if len(params) != 0:
            torch.orig_save(obj, f, params)
        else:
            torch.orig_save(obj, f)

    def _prepare_pruners(self, model: torch.nn.Module, opt: torch.optim.Optimizer):
        """Wrapper the model and optimizer to support all the pruning functionality.

        :param config: WeightPruningConfig
        :param model: The user's model, a torch.nn.Module object
        :param opt: The user's optimizer, a torch.optim object
        :return: The modified model and optimizer
        """
        import torch
        torch.orig_save = torch.save  ##rewrite torch save
        setattr(torch, 'save', self.save)
        model.pruners = self.pruners
        opt.pruners = self.pruners

        inc_hook_handle = self._register_on_step_begin(model)
        model.inc_hook_handle = inc_hook_handle
        self._rewrite_optimizer_step(opt)
        # return model, opt

    # def complete_pruning(model: torch.nn.Module, opt: torch.optim):
    #     """UnWrapper the model and optimizer
    #     :param model: the modified model
    #     :param opt: the modified optimizer
    #     :return: the pruned model and the user's optimizer
    #     """
    #
    #     model.inc_hook_handle.remove()
    #     delattr(model, "inc_hook_handle")
    #     model.pruners = None
    #     delattr(model, "pruners")
    #     opt.pruners = None
    #     delattr(opt, "pruners")
    #     opt.step = opt.orig_step
    #     delattr(opt, "orig_step")
    #     return model, opt


@register_pruning("sparse_gpt_pruning")
class SparseGPTPruning(BasePruning):
    def __init__(self, config, model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 framework ='pytorch', device: str =None):
        """Initialize."""
        super().__init__(config, model)
        assert 'cpu' in device or 'cuda' in device, f"Only cpu and cuda are supported."
        self.dev = torch.device(device)
        self._layers = []
        self.collect_inputs = []
        self._dataloader = dataloader
        self._prepare_pruners(model, dataloader)
        
    def _prepare_pruners(self, model: torch.nn.Module,
                        dataloader: torch.utils.data.DataLoader):
        """Wrapper the model and optimizer to support all the pruning functionality.

        :param config: WeightPruningConfig
        :param model: The user's model, a torch.nn.Module object
        :param opt: The user's optimizer, a torch.optim object
        :return: The modified model and optimizer
        """
        self.dev = torch.device(type='cpu')
        self.model_dev = self._model.device
        self._layers = get_layers(self._model)
        if torch.cuda.is_available():
            
            self.dev = torch.device(type='cuda')
        self._do_pruning(self._layers)
        self._model = self._model.to(self.model_dev)
        # TODO add get_sparsity_ratio() for sparseGPT

    @torch.no_grad()
    def _do_pruning(self, layers):
        self._model = self._model.cpu()
        
        inputs, inp_dict = collect_layer_inputs(model=self._model, layers=layers, layer_idx=0, \
                                                layer_inputs=self._dataloader, device=self.dev)
        if 'cuda' in self.dev.type:
            torch.cuda.empty_cache()
        
        for i in tqdm(range(len(layers))):
            layer = layers[i].to(self.dev)
            layer_index_str = '.' + str(i) + '.'
            handles_list = []
            for pruner in self.pruners:
                layer_op_names = [key for key in pruner.modules.keys() if layer_index_str in key]
                handles_list.append(pruner.register_gpt_hook(layer_op_names))
            for j in range(len(inputs)):
                layer(inputs[j], **inp_dict)[0]
            for handles in handles_list:
                for h in handles:
                    h.remove()
            for pruner in self.pruners:
                layer_op_names = [key for key in pruner.modules.keys() if layer_index_str in key]
                pruner.fasterprune(layer_op_names)
            for j in range(len(inputs)):
                # the weights of current layer have been pruned, get the latest outputs as the inputs for next layer
                inputs[j] = layer(inputs[j], **inp_dict)[0]
            layers[i] = layer.cpu()
            if 'cuda' in self.dev.type:
                torch.cuda.empty_cache()
    
    
@register_pruning("retrain_free_pruning")
class RetrainFreePruning(BasePruning):
    def __init__(self, config, model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_func = None,
                 framework ='pytorch'):
        """Initialize."""
        super().__init__(config, model)
        # self.dev = model.device
        self._dataloader = dataloader
        self._loss_func = loss_func
        self._prepare_pruners()
        
    def _register_on_step_begin(self, model: torch.nn.Module, pruners):
        """Mount on_step_begin to the model.

        :param model:The torch module to be pruned
        :return: hook handle
        """

        def hook(module, input):
            for pruner in pruners:
                pruner.on_step_begin(0)

        hook_handle = model.register_forward_pre_hook(hook)
        return hook_handle
    
    # def _register_on_step_end(self, model: torch.nn.Module, pruners):
    #     """Mount on_step_begin to the model.

    #     :param model:The torch module to be pruned
    #     :return: hook handle
    #     """
    #     def hook(_, grad_in, grad_out):
    #         for pruner in pruners:
    #             pruner.on_step_end()
    #     hook_handle = model.register_backward_hook(hook)
    #     return hook_handle
        
    def _prepare_pruners(self):
        """Wrapper the model and optimizer to support all the pruning functionality.

        :param config: WeightPruningConfig
        :param model: The user's model, a torch.nn.Module object
        :param opt: The user's optimizer, a torch.optim object
        :return: The modified model and optimizer
        """
        hook_handle_before = self._register_on_step_begin(self._model, self.pruners)
        self._do_pruning()
        hook_handle_before.remove()
        get_sparsity_ratio(self.pruners, self._model)

    def _do_pruning(self):
        progress_bar = tqdm(range(len(self._dataloader)))
        if self._loss_func is not None:
            for inputs, target in self._dataloader:
                outputs = self._model(inputs)
                loss = self._loss_func(outputs, target)
                loss.backward()
                for pruner in self.pruners:
                    pruner.on_step_end()
                progress_bar.update(1)
        else:
            for batch in self._dataloader:
                outputs = self._model(return_dict=True, **batch)
                loss = outputs.loss
                loss.backward()
                for pruner in self.pruners:
                    pruner.on_step_end()
                progress_bar.update(1)
        
        
