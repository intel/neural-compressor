import os
import json
import psutil

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from .torch_load import load


def get_children(model):
    module_list = []
    children = list(model.children())
    if len(children) == 0:
        return [model]
    for child in children:
        module_list += get_children(child)
    return module_list


def get_named_children(model, pre=[]):
    module_list = []
    if len(list(model.children())) == 0:
        return [('.'.join(pre), model)]
    for name, module in model.named_children():
        module_list += get_named_children(module, pre=pre + [name])
    return module_list


def load_shell(path, cls):
    if cls.__base__ == _BaseAutoModelClass:
        config = AutoConfig.from_pretrained(path)
        with init_empty_weights():
            model = cls.from_config(config)
    else:
        config = cls.config_class.from_pretrained(path)
        with init_empty_weights():
            model = cls(config)
    model.tie_weights()
    model.eval()
    return model


def get_module_by_name(model, module_name):
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)
        else:
            return None
    if hasattr(model, name_list[-1]):
        return model
    else:
        return None


def update_module(model, module_name, new_module):
    super_module = get_module_by_name(model, module_name)
    if super_module:
        setattr(super_module, module_name.split('.')[-1], new_module)


def load_layer_wise_quantized_model(path):
    model = torch.load(os.path.join(path, 'model_arch.pt'))
    for name, _ in model.named_modules():
        if name + '.pt' in os.listdir(path):
            update_module(model, name, torch.load(os.path.join(path, name + '.pt')))
    model.eval()
    return model


def load_from_shard(path, tensor_name):
    idx_dict = json.load(open(os.path.join(path, 'pytorch_model.bin.index.json'), 'r'))['weight_map']
    assert tensor_name in idx_dict.keys(), '{} not the index.json'.format(tensor_name)
    state_dict = torch.load(os.path.join(path, idx_dict[tensor_name]))
    return state_dict[tensor_name]


def load_tensor_from_shard(path, tensor_name):
    idx_dict = json.load(open(os.path.join(path, 'pytorch_model.bin.index.json'), 'r'))['weight_map']
    assert tensor_name in idx_dict.keys(), '{} not the index.json'.format(tensor_name)
    return load_tensor(os.path.join(path, idx_dict[tensor_name]), tensor_name)


def load_tensor(path, tensor_name=None):
    if 'lm_head' in tensor_name:
        tensor_name = 'model.decoder.embed_tokens.weight'
    state_dict = load(path, tensor_name)
    # return state_dict
    return state_dict[tensor_name]


def get_memo():
    process = psutil.Process(os.getpid())
    memo = float(process.memory_full_info().uss/(1024*1204))
    return memo
