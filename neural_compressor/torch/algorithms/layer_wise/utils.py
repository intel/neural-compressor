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

import gc
import json
import os

import torch
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from safetensors.torch import save_file

from neural_compressor.common import options
from neural_compressor.torch.algorithms.weight_only.modules import INCWeightOnlyLinear
from neural_compressor.torch.utils import is_hpu_available
from neural_compressor.torch.utils.utility import dowload_hf_model

from .load import load

LWQ_WORKSPACE = os.path.join(options.workspace, "lwq_tmpdir")


class QDQLayer(torch.nn.Module):
    """Quantized and Dequantized Layer."""

    def __init__(self, module, input_scale=None) -> None:
        """Init the QDQLayer object."""
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.module = module
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.input_scale = input_scale

    def forward(self, X):
        """Forward function."""
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


def load_tensor_from_safetensors(path, tensor_name=None, prefix=None, device="cpu"):
    """Load a tensor from safetensors file with given tensor name."""
    with safe_open(path, framework="pt", device=device) as f:
        if tensor_name in f.keys():
            value = f.get_tensor(tensor_name)
        elif prefix and tensor_name.replace(f"{prefix}.", "") in f.keys():
            value = f.get_tensor(tensor_name.replace(f"{prefix}.", ""))
        else:
            raise ValueError(f"Tensor '{tensor_name}' not found in the file '{path}'")
    return value


def load_tensor_from_safetensors_shard(
    pretrained_model_name_or_path, tensor_name, prefix=None, device="cpu"
):  # pragma: no cover
    """Load tensor from shard."""
    path = _get_path(pretrained_model_name_or_path)
    idx_dict = json.load(open(os.path.join(path, "model.safetensors.index.json"), "r"))["weight_map"]
    if tensor_name not in idx_dict.keys():
        if tensor_name.replace(f"{prefix}.", "") in idx_dict.keys():
            tensor_name = tensor_name.replace(f"{prefix}.", "")
        else:
            assert False, "{} not in the index.json".format(tensor_name)
    return load_tensor_from_safetensors(os.path.join(path, idx_dict[tensor_name]), tensor_name, device)


def _get_path(pretrained_model_name_or_path):
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:  # pragma: no cover
        path = pretrained_model_name_or_path
    else:
        path = dowload_hf_model(pretrained_model_name_or_path)
    return path


get_path = _get_path


def load_value(model, param_name, path, device="cpu"):
    """Load the module value.

    Args:
        model (torch.nn.module): torch model.
        param_name (str): module name.
        path (str): path to load state_dict per layer.
        device (str, optional): module device. Defaults to "cpu".

    Returns:
        tensor: the module value.
    """
    if "lm_head" in param_name and getattr(model.config, "tie_word_embeddings", True):
        input_embeddings = model.get_input_embeddings()
        modules = get_named_children(model)
        for name, module in modules:
            if module == input_embeddings:
                param_name = name + "." + param_name.split(".")[-1]
    prefix = model.base_model_prefix
    files = os.listdir(path)
    safetensors_files = [filename for filename in files if filename.endswith(".safetensors")]

    if device == torch.device("hpu"):
        device = "hpu"

    if len(safetensors_files) == 1:
        value = load_tensor_from_safetensors(
            os.path.join(path, "model.safetensors"), param_name, prefix=prefix, device=device
        )
    elif len(safetensors_files) >= 2:
        value = load_tensor_from_safetensors_shard(path, param_name, prefix=prefix, device=device)
    elif "pytorch_model.bin.index.json" in files:
        value = load_tensor_from_shard(path, param_name, prefix)
    else:
        value = load_tensor(os.path.join(path, "pytorch_model.bin"), param_name, prefix)
    return value


def load_module(model, module_name, path, device="cpu"):
    """Load all named parameters of module.

    Args:
        model (torch.nn.module): torch model.
        module_name (str): module name.
        path (str): path to load state_dict per layer.
        device (str, optional): module device. Defaults to "cpu".
    """
    module = get_module(model, module_name)
    for n, p in module.named_parameters():
        param_name = module_name + "." + n
        value = load_value(model, param_name, path, device)
        set_module_tensor_to_device(model, param_name, device, value)


def load_first_layer_only(user_model, model_name):
    """Load first layer only.

    Args:
        user_model (torch.nn.Module): input model
        model_name (str): model name or path
    """
    for name, m in user_model.named_modules():
        if ("layers" not in name or "layers.0" in name) and len(name) > 0 and len(list(m.named_children())) == 0:
            load_module(user_model, name, get_path(model_name), device="hpu" if is_hpu_available() else "cpu")


def register_weight_hooks(model, path, device="cpu", clean_weight=True, saved_path=None, indicated_layers=None):
    """Register weight hooks for model.

    Args:
        model (torch.nn.module): torch model.
        path (str): path to load state_dict per layer.
        device (str, optional): module device. Defaults to "cpu".
        clean_weight (bool, optional): to clean model weight. Defaults to True.
        saved_path (str, optional): path to save module weight. Defaults to None.

    Returns:
        list: handlers.
    """
    if saved_path:
        os.makedirs(saved_path, exist_ok=True)

    def forward_pre_hook(name):
        def hook(module, input):
            state_dict = None
            if os.path.exists(os.path.join(LWQ_WORKSPACE, f"{name}.pt")):
                state_dict = torch.load(
                    os.path.join(LWQ_WORKSPACE, f"{name}.pt"),
                    map_location=torch.device(device) if isinstance(device, str) else device,
                )
            for n, p in module.named_parameters():
                param_name = name + "." + n
                if state_dict:
                    value = state_dict[n]
                else:
                    value = load_value(model, param_name, path, device=device)
                set_module_tensor_to_device(model, param_name, device, value)
            module = module.to(device)

        return hook

    def forward_hook(name):
        def hook(module, input, output):
            if saved_path:
                file_path = os.path.join(saved_path, f"{name}.pt")
                torch.save(module.state_dict(), file_path)
            clean_module_weight(module)

        return hook

    handle = {}
    modules = get_named_children(model)
    for name, module in modules:
        if indicated_layers is not None and name not in indicated_layers:  # pragma: no cover
            # load other layers to memory
            state_dict = None
            if os.path.exists(os.path.join(LWQ_WORKSPACE, f"{name}.pt")):
                state_dict = torch.load(
                    os.path.join(LWQ_WORKSPACE, f"{name}.pt"),
                    map_location=torch.device(device) if isinstance(device, str) else device,
                )
            for n, p in module.named_parameters():
                param_name = name + "." + n
                if state_dict:
                    value = state_dict[n]
                else:
                    value = load_value(model, param_name, path, device=device)
                set_module_tensor_to_device(model, param_name, device, value)
            module = module.to(device)
            continue
        handle[name] = [module.register_forward_pre_hook(forward_pre_hook(name))]
        if clean_weight:
            handle[name] += [module.register_forward_hook(forward_hook(name))]
    return handle


def clean_module_weight(module):
    """Clean module weight."""
    if isinstance(module, QDQLayer):
        submodule = module.module
    else:
        submodule = module

    if isinstance(module, INCWeightOnlyLinear):
        for n, m in submodule._buffers.items():
            old_value = getattr(submodule, n)
            with torch.no_grad():
                submodule._buffers[n] = torch.zeros(old_value.shape, device="meta")

    for n, m in submodule.named_parameters():
        is_buffer = n in submodule._buffers
        old_value = getattr(submodule, n)
        with torch.no_grad():
            if is_buffer:
                submodule._buffers[n] = torch.zeros(old_value.shape, device="meta")
            else:
                param_cls = type(submodule._parameters[n])
                kwargs = submodule._parameters[n].__dict__
                if is_hpu_available():
                    from habana_frameworks.torch.core import weight_sharing

                    if param_cls == weight_sharing.HabanaParameterWrapper:
                        try:
                            kwargs.pop("change_device_placement")
                        except KeyError:
                            pass

                new_value = torch.zeros(old_value.shape, device="meta")
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to("meta")
                submodule._parameters[n] = new_value
    # gc.collect()


def save_layers_in_shards_iteratively(checkpoint_dir, output_dir, layers_per_shard=10):
    """Save model layers iteratively in shards, each shard containing a fixed number of layers using safetensors."""
    os.makedirs(output_dir, exist_ok=True)

    # Get list of checkpoint files in the checkpoint_dir
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    checkpoint_files.sort()

    bin_index = {}
    shard_idx = 0
    layer_buffer = []

    # Iterate over each checkpoint file and process layer state dicts
    for checkpoint_file in checkpoint_files:
        layer_path = os.path.join(checkpoint_dir, checkpoint_file)
        print(f"Loading layer from {layer_path}")

        # Load the layer checkpoint
        checkpoint = torch.load(layer_path, map_location="cpu")
        layer_state_dict = checkpoint

        # Add the layer's state dict to the buffer
        layer_buffer.append(layer_state_dict)

        # If the buffer has reached the number of layers per shard, save the shard
        if len(layer_buffer) == layers_per_shard:
            shard_state_dict = {}
            for i, layer_dict in enumerate(layer_buffer):
                shard_state_dict.update(layer_dict)
                # Update the bin index for each layer
                for layer_name in layer_dict.keys():
                    bin_index[layer_name] = shard_idx

            # Save the shard to disk using safetensors
            shard_filename = f"model_shard-{str(shard_idx + 1).zfill(5)}-of-{str((len(checkpoint_files) // layers_per_shard) + 1).zfill(5)}.safetensors"
            shard_path = os.path.join(output_dir, shard_filename)
            save_file(shard_state_dict, shard_path)  # Save using safetensors
            print(f"Saved shard {shard_idx + 1} of {len(checkpoint_files) // layers_per_shard + 1} at {shard_path}")

            # Clear the buffer and move to the next shard
            layer_buffer.clear()
            shard_idx += 1

    # If there are any remaining layers in the buffer, save them as the last shard
    if layer_buffer:
        shard_state_dict = {}
        for i, layer_dict in enumerate(layer_buffer):
            shard_state_dict.update(layer_dict)
            # Update the bin index for each layer
            for layer_name in layer_dict.keys():
                bin_index[layer_name] = shard_idx

        # Save the final shard
        shard_filename = f"model_shard-{str(shard_idx + 1).zfill(5)}-of-{str((len(checkpoint_files) // layers_per_shard) + 1).zfill(5)}.safetensors"
        shard_path = os.path.join(output_dir, shard_filename)
        save_file(shard_state_dict, shard_path)  # Save using safetensors
        print(f"Saved final shard {shard_idx + 1} of {len(checkpoint_files) // layers_per_shard + 1} at {shard_path}")

    # Save bin index to a JSON file
    bin_index_file = os.path.join(output_dir, "model_bin_index.json")
    with open(bin_index_file, "w") as f:
        json.dump(bin_index, f, indent=4)

    print(f"Model bin index saved to {bin_index_file}")


from safetensors.torch import load_file  # Safetensors load function


def load_model_from_shards_with_safetensors(shard_dir, bin_index_file):
    """Load the model from its shards and the bin index using safetensors.

    Args:
        shard_dir (str): Directory containing the model shard files.
        bin_index_file (str): Path to the bin index JSON file.

    Returns:
        torch.nn.Module: The reconstructed model with the layers.
    """
    # Load bin index to get the layer -> shard mapping
    with open(bin_index_file, "r") as f:
        bin_index = json.load(f)

    full_state_dict = {}

    # Sort and load the shard files
    shard_files = [f for f in os.listdir(shard_dir) if f.endswith(".safetensors")]
    shard_files.sort()

    for shard_file in shard_files:
        shard_path = os.path.join(shard_dir, shard_file)
        print(f"Loading shard from {shard_path}")
        shard_state_dict = load_file(shard_path, device="hpu" if is_hpu_available() else "cpu")
        full_state_dict.update(shard_state_dict)

    return full_state_dict
