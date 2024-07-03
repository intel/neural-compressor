#!/usr/bin/env python
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
"""Utils for layer wise quantization."""

import os
import gc
import json
import pickle
from functools import partial
import logging
from collections import OrderedDict

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from .load import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(filename)s L%(lineno)d: %(message)s")
logger = logging.getLogger("layer_wise_tools")

LWQ_WORKSPACE = os.path.join("layer_wise_tmp")


class QDQLayer(torch.nn.Module):
    def __init__(self, module, input_scale=None) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.module = module
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.input_scale = input_scale

    def forward(self, X):
        if self.input_scale is not None:
            X = torch.mul(X, self.input_scale)
        X = self.quant(X)
        X = self.module(X)
        X = self.dequant(X)
        return X


def get_module(model, key):
    """Get module from model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    attrs = key.split(".")
    module = model
    for attr in attrs:
        try:
            attr = int(attr)
            module = module[attr]
        except:
            module = getattr(module, attr)
    return module


def get_children(model):
    """Get all the children of given model."""
    module_list = []
    children = list(model.children())
    if len(children) == 0:
        return [model]
    for child in children:
        module_list += get_children(child)
    return module_list


def get_named_children(model, pre=[]):
    """Get all the name and children of given model."""
    module_list = []
    if len(list(model.children())) == 0:
        return [(".".join(pre), model)]
    for name, module in model.named_children():
        module_list += get_named_children(module, pre=pre + [name])
    return module_list


def dowload_hf_model(repo_id, cache_dir=None, repo_type=None, revision=None):
    """Download hugging face model from hf hub."""
    from huggingface_hub.constants import DEFAULT_REVISION, HUGGINGFACE_HUB_CACHE
    from huggingface_hub.file_download import REGEX_COMMIT_HASH, repo_folder_name
    from huggingface_hub.utils import EntryNotFoundError

    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if revision is None:
        revision = DEFAULT_REVISION
    if repo_type is None:
        repo_type = "model"
    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    commit_hash = None
    if REGEX_COMMIT_HASH.match(revision):
        commit_hash = revision
    else:
        ref_path = os.path.join(storage_folder, "refs", revision)
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                commit_hash = f.read()
    if storage_folder and commit_hash:
        pointer_path = os.path.join(storage_folder, "snapshots", commit_hash)
        if os.path.isdir(pointer_path):
            return pointer_path
    else:  # pragma: no cover
        from huggingface_hub import snapshot_download

        file_path = snapshot_download(repo_id)
        return file_path


def load_empty_model(pretrained_model_name_or_path, cls=AutoModelForCausalLM, save_path=None, **kwargs):
    """Load a empty model."""
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:  # pragma: no cover
        path = pretrained_model_name_or_path
    else:
        path = dowload_hf_model(pretrained_model_name_or_path)
    if cls.__base__ == _BaseAutoModelClass:
        config = AutoConfig.from_pretrained(path, **kwargs)
        with init_empty_weights():
            model = cls.from_config(config)
    else:  # pragma: no cover
        config = cls.config_class.from_pretrained(path, **kwargs)
        with init_empty_weights():
            model = cls(config)
    model.tie_weights()
    model.eval()
    model.path = pretrained_model_name_or_path

    if save_path is None:
        save_path = LWQ_WORKSPACE
    convert_model(model, save_path)
    return model


def get_super_module_by_name(model, module_name):
    """Get the father module with given name of child module."""
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)
        else:  # pragma: no cover
            return None
    if hasattr(model, name_list[-1]):
        return model
    else:  # pragma: no cover
        return None


def update_module(model, module_name, new_module):
    """Update module."""
    super_module = get_super_module_by_name(model, module_name)
    if super_module:
        setattr(super_module, module_name.split(".")[-1], new_module)


def get_layers_before_block(model):
    """get the embed layers before blocks."""
    return_layers = []
    block_name = None
    def _forward(module, name, *args, **kwargs):
        if name == block_name:
        # if 'DecoderLayer' in name:
            raise NotImplementedError
        if len(module._modules) == 0:
            return_layers.append((name, module))
        return module.ori_forward(*args, **kwargs)

    for n, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList):
            block_name = n + '.' + m.named_children().__next__()[0]
        m.ori_forward = m.forward
        m.forward = partial(_forward, m, n)
    
    try:
        model.forward(
            input_ids=torch.zeros((1,1), device='meta', dtype=torch.int),
            attention_mask=torch.zeros((1,1), device='meta', dtype=torch.int)
            )
    except NotImplementedError:
        pass

    for n, m in model.named_modules():
        m.forward = m.ori_forward
        del m.ori_forward
    
    return return_layers



def load_layer_wise_quantized_model(path):  # pragma: no cover
    """Load layer wise quantized model."""
    model = torch.load(os.path.join(path, "model_arch.pt"))
    for name, _ in model.named_modules():
        if name + ".pt" in os.listdir(path):
            update_module(model, name, torch.load(os.path.join(path, name + ".pt")))
    model.eval()
    return model


def load_tensor_from_shard(pretrained_model_name_or_path, tensor_name, prefix=None):  # pragma: no cover
    """Load tensor from shard."""
    path = _get_path(pretrained_model_name_or_path)
    idx_dict = json.load(open(os.path.join(path, "pytorch_model.bin.index.json"), "r"))["weight_map"]
    if tensor_name not in idx_dict.keys():
        if tensor_name.replace(f"{prefix}.", "") in idx_dict.keys():
            tensor_name = tensor_name.replace(f"{prefix}.", "")
        else:
            assert False, "{} not in the index.json".format(tensor_name)
    return load_tensor(os.path.join(path, idx_dict[tensor_name]), tensor_name, None)


def load_tensor(path, tensor_name=None, prefix=None):
    """Load a tensor from bin file with given tensor name."""
    # transformers.modeling_utils
    if tensor_name:
        if "gamma" in tensor_name:  # pragma: no cover
            tensor_name = tensor_name.replace("gamma", "weight")
        if "beta" in tensor_name:  # pragma: no cover
            tensor_name = tensor_name.replace("beta", "bias")

    if os.path.isdir(path):
        path = os.path.join(path, "pytorch_model.bin")
    state_dict = load(path, tensor_name, prefix)
    if tensor_name:
        if tensor_name in state_dict:
            return state_dict[tensor_name]
        else:  # pragma: no cover
            return state_dict[tensor_name.replace(f"{prefix}.", "")]
    else:  # pragma: no cover
        return state_dict


def _get_path(pretrained_model_name_or_path):
    if pretrained_model_name_or_path is None:
        return None
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:  # pragma: no cover
        path = pretrained_model_name_or_path
    else:
        path = dowload_hf_model(pretrained_model_name_or_path)
    return path


def load_value(model, param_name, path):
    logger.debug(f'load value for layer: {param_name}')
    if "lm_head" in param_name and getattr(model.config, "tie_word_embeddings", True):
        input_embeddings = model.get_input_embeddings()
        modules = get_named_children(model)
        for name, module in modules:
            if module == input_embeddings:
                param_name = name + "." + param_name.split(".")[-1]
    prefix = model.base_model_prefix
    if "pytorch_model.bin.index.json" in os.listdir(path):
        value = load_tensor_from_shard(path, param_name, prefix)
    else:
        value = load_tensor(os.path.join(path, "pytorch_model.bin"), param_name, prefix)
    return value


def load_module(model, module_name, path, device="cpu"):
    module = get_module(model, module_name)
    for n, p in module.named_parameters():
        param_name = module_name + "." + n
        value = load_value(model, param_name, path)
        set_module_tensor_to_device(model, param_name, device, value)


def register_weight_hooks(model, path, device="cpu", clean_weight=True, saved_path=None):
    if saved_path:
        os.makedirs(saved_path, exist_ok=True)

    def forward_pre_hook(name):
        def hook(module, input):
            logger.debug(f"{name} forward hood load value")
            state_dict = None
            if os.path.exists(os.path.join(saved_path, f"{name}.pt")):
                state_dict = torch.load(os.path.join(saved_path, f"{name}.pt"))
            for n, p in module.named_parameters():
                param_name = name + "." + n
                if state_dict:
                    value = state_dict[n]
                else:
                    value = load_value(model, param_name, path)
                set_module_tensor_to_device(model, param_name, device, value)
            module = module.to(device)
            
        return hook

    def forward_hook(name):
        def hook(module, input, output):
            logger.debug(f"{name} forward hood clean value")
            if saved_path:
                file_path = os.path.join(saved_path, f"{name}.pt")
                torch.save(module.state_dict(), file_path)
            clean_module_weight(module)

        return hook

    handle = {}
    modules = get_named_children(model)
    for name, module in modules:
        handle[name] = [module.register_forward_pre_hook(forward_pre_hook(name))]
        if clean_weight:
            handle[name] += [module.register_forward_hook(forward_hook(name))]
    return handle


def clean_module_weight(module):
    if isinstance(module, QDQLayer):
        submodule = module.module
    else:
        submodule = module

    for n, m in submodule.named_parameters():
        is_buffer = n in submodule._buffers
        old_value = getattr(submodule, n)
        with torch.no_grad():
            if is_buffer:
                submodule._buffers[n] = torch.zeros(old_value.shape, device="meta")
            else:
                param_cls = type(submodule._parameters[n])
                kwargs = submodule._parameters[n].__dict__
                new_value = torch.zeros(old_value.shape, device="meta")
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to("meta")
                submodule._parameters[n] = new_value
    gc.collect()


def convert_model(empty_model, saved_path=LWQ_WORKSPACE):
    def _get_value(name, n):
        state_dict = None
        if os.path.exists(os.path.join(saved_path, f"{name}.pt")):
            state_dict = torch.load(os.path.join(saved_path, f"{name}.pt"))
        param_name = name + "." + n
        if state_dict:
            value = state_dict[n]
        else:
            value = load_value(empty_model, param_name, empty_model.path)
        return value

    def _update(module):
        state_dict = None
        if os.path.exists(os.path.join(saved_path, f"{name}.pt")):
            state_dict = torch.load(os.path.join(saved_path, f"{name}.pt"))
        for n, p in module.named_parameters():
            if str(p.device) != 'meta':
                continue
            param_name = name + "." + n
            if state_dict:
                value = state_dict[n]
            else:
                value = load_value(empty_model, param_name, saved_path)
            set_module_tensor_to_device(empty_model, param_name, 'cpu', value)
        file_path = os.path.join(saved_path, f"{name}.pt")
        torch.save(module.state_dict(), file_path)

    def _layer_wise_to(module, name, device_or_dtype):
        if isinstance(device_or_dtype, torch.dtype):
            return module.ori_to(device_or_dtype)
        elif len(module._modules) == 0:
            # skip method type
            if len(module._parameters) == 0 or module.weight.device.type != 'meta':
                return module.ori_to(device_or_dtype)
            else:
                for n, _ in module.named_parameters():
                    param_name = name + "." + n
                    value = load_value(empty_model, param_name, empty_model.path)
                    dtype = None
                    if hasattr(module, 'dtype'):
                        dtype = module.dtype
                    set_module_tensor_to_device(module, n, device_or_dtype, value, dtype=dtype)
                return module.ori_to(device_or_dtype)
        else:
            for n, m in module.named_children():
                m.to(device_or_dtype)
            return module

    modules = get_named_children(empty_model)
    for name, module in modules:
        if hasattr(module, 'weight'):
            # delattr(module, 'weight')
            # module.weight = partial(_get_value, name, 'weight')()
            module.get_weight = partial(_get_value, name, 'weight')
        if hasattr(module, 'bias') and module.bias is not None:
            module.get_bias = partial(_get_value, name, 'bias')
        module.update = partial(_update, module)
    
    def _repalce_to(module, name):
        if len(module._modules) > 0:
            for n, m in module.named_children():
                if len(name) > 0:
                    n = name + '.' + n
                _repalce_to(m, n)
        module.ori_to = module.to
        module.to = partial(_layer_wise_to, module, name)
    _repalce_to(empty_model, '')

def load_model_with_hooks(
        pretrained_model_name_or_path,
        cls=AutoModelForCausalLM,
        device="cpu",
        clean_weight=True,
        saved_path=None, 
        **kwargs):
    if saved_path is None:
        saved_path = LWQ_WORKSPACE
    empty_model = load_empty_model(pretrained_model_name_or_path, cls=cls, **kwargs)
    register_weight_hooks(empty_model, empty_model.path, device, clean_weight, saved_path)
    return empty_model


def layer_wise_save(model, path):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, 'layer_wise_model.bin')
    modules = get_named_children(model)
    with open(file_path, 'wb') as f:
        for name, module in modules:
            output = OrderedDict()
            if hasattr(module, "get_weight"):
                output[f"{name}.weight"] = module.get_weight()
            if hasattr(module, "get_bias"):
                output[f"{name}.bias"] = module.get_bias()
            output = pickle.dumps(output)
            f.write(output + b'split_tag')

def layer_wise_load(path):
    file_path = os.path.join(path, 'layer_wise_model.bin')
    state_dict = OrderedDict()
    data = open(file_path, 'rb').read().split(b'split_tag')
    for d in data:
        if len(d) > 0:
            d = pickle.loads(d)
            state_dict.update(d)