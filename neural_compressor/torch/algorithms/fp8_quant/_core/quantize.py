# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import habana_frameworks.torch.core as htcore
import torch
import torch.nn as nn
import numpy as np

from .._quant_common.quant_config import QuantMode
from .._quant_common.helper_modules import PatchedUnmeasuredModule
from .._quant_common.quant_config import get_hqt_config, set_hqt_config
from ..utils.logger import logger
from .common import generate_model_info, mod_default_dict, parent_child_mod_dict, \
                    save_scales, load_scales
from .measure import load_measurements
from .scale import scale_method_mapping, scaling_methods, convert_scales_to_tensors_dict, load_layer_scales
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator
cur_accelerator = auto_detect_accelerator()


def patch_module(mod, qconfig, mod_dict, patched_mod=None):
    """Replaces the module with patched module according to mod_dict.

    Args:
        mod (nn.module): The module that will be replaced with a patched module that quantize the inputs/outputs.
        qconfig (ModuleExtraConfig): The quantization config object with the information how to quantize the inputs/outputs.
        mod_dict (dict): dictionary from module name to its patched module.

    Returns:
        nn.module: The new patched module after patching.
    """
    parent = parent_child_mod_dict[mod].parent
    name = parent_child_mod_dict[mod].name
    if patched_mod is None:
        patched_mod = mod_dict[mod.__class__.__name__].patched_module(mod, qconfig)
    setattr(parent, name, patched_mod)


def apply_hf_hook(module):
    """Applies hf_hook on a given module so its weights will be loaded from disk to cpu and then we can quantize it."""
    if hasattr(module, "_hf_hook"):
        module._hf_hook.pre_forward(module)
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")
    if hasattr(module, "_old_forward"):
        module.forward = module._old_forward
        delattr(module, "_old_forward")


def quantize_params(mod, mod_extra_config):
    """Quantizes the weights of the given module according to the quantization info from mod_extra_config.

    Args:
        mod (nn.module): The module that its weights will be quantized.
        mod_extra_config (ModuleExtraConfig): The quantization config object with the information how to quantize the inputs/outputs.
    """
    for param_name in mod_extra_config.params:
        quantizer = mod_extra_config.params[param_name][0]
        param = getattr(mod, param_name)
        if param.dtype == torch.float16:
            param = param.to(torch.bfloat16)
        quantized_param = quantizer(param.to(cur_accelerator.name()))
        delattr(mod, param_name)
        setattr(mod, param_name, nn.Parameter(quantized_param))
        quantized_param = getattr(mod, param_name)
        quantized_param.requires_grad_(False)
        cur_accelerator.synchronize()


def convert_fp16_to_bf16(model):
    """Convert all float16 parameters and buffers in the model to bfloat16 after FP8 quantization.
    
    Args:
        model (torch.nn.Module): The PyTorch model that needs to be converted.
    """
    # convert parameters
    for name, param in model.named_parameters():
        if param.dtype == torch.float16:
            param.data = param.data.to(torch.bfloat16)
            logger.debug("Convert FP16 to BF16, parameter name: %s", name)
    
    # convert buffers
    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.float16:
            buffer.data = buffer.data.to(torch.bfloat16)
            logger.debug("Convert FP16 to BF16, buffer name: %s", name)


def prepare_model(model, mod_list, measurement, scale_file, scaling_method, scale_config):
    """Calculates scales according to the scaling method and config.
    Replaces the model submodules according to the mod_list with patched quantization modules.
    Configures patched modules with the quantization/dequantization methods to apply on their input and output tensors.
    Quantizes the model parameters as they are static.

    Args:
        model (nn.module): The model to quantize.
        mod_list (list): The specific submodules that will be quantized in the model.
        measurement (dict): The measurements of the model.
        scale_file (str): The file containing the scales.
        scaling_method (str): The scaling method to use.
        scale_config (dict): The scaling configuration.
    """
    config = get_hqt_config(model)
    recalc_scales = config.cfg["recalc_scales"]
    scales_file_format = np.ndarray
    scales_obj = (
        load_scales(scale_file + ".npz", scales_file_format)
        if (scale_file is not None) and not recalc_scales
        else {}
    )
    scales = convert_scales_to_tensors_dict(scales_obj, scales_file_format, scale_config["hp_dtype"])
    save_file = False
    patched_modules = []
    patched_module_types = set()
    with torch.no_grad():
        for name, mod in model.named_modules():
            mod_type_str = mod.__class__.__name__
            if name in mod_list and name not in scales and config.cfg["use_stats_files"] and name not in measurement:
                if mod_default_dict[mod_type_str].should_measure_and_quant:
                    if not config.cfg["ignore_modules_wo_measures"]:
                        patch_module(mod, None, None, PatchedUnmeasuredModule(name))
                    else:
                        logger.debug("Module %s was not quantized.", name)
                    continue
            # When offloading weight to disk, need to transfer the weight from disk to cpu using hf_hook
            apply_hf_hook(mod)
            if name in mod_list:
                set_hqt_config(mod, config)  # set config in the module, as it consumed by the patched module
                mod_extra_config, save_file = load_layer_scales(mod, name, config,
                                                                mod_type_str, measurement,
                                                                scales, scale_file,
                                                                scales_file_format,
                                                                scales_obj, scaling_method,
                                                                scale_config, save_file)
                if not config.cfg["fake_quant"] and mod_default_dict[mod_type_str].should_measure_and_quant:
                    quantize_params(mod, mod_extra_config)
                patch_module(mod, mod_extra_config, mod_default_dict)
                patched_modules.append(name)
                patched_module_types.add(type(mod))
                logger.debug("Patched module name: %s", name)
    if save_file: # cache calculated scales
        save_scales(model, scales_obj, scales_file_format, scale_file + ".npz")
        save_scales(model, scales_obj, scales_file_format, scale_file + ".json")
    logger.debug("Patched module types: %s", patched_module_types)
    logger.debug("Patched modules: %s", patched_modules)
    logger.debug("Total patched modules: %d", len(patched_modules))
    model = model.to(cur_accelerator.name())
    convert_fp16_to_bf16(model)
    cur_accelerator.synchronize()


def prepare_model_with_dummy_measurement(model, mod_list, scaling_method, scale_config):
    """Aim for loading, replace module with patched module for model on meta device.

    Args:
        model (torch.nn.Module): empty model on meta device
        mod_list (list): The specific submodules that will be quantized in the model.
        scaling_method (str): The scaling method to use.
        scale_config (dict): The scaling configuration.

    Returns:
        model: empty model that quantized by default qconfig.
    """
    from .common import ModuleConfig, ModuleExtraConfig, mod_types

    config = get_hqt_config(model)
    patched_modules = []
    patched_module_types = set()
    with torch.no_grad():
        for name, mod in model.named_modules():
            if name not in mod_list:
                continue
            set_hqt_config(mod, config)  # set config in the module, as it consumed by the patched module
            mod_type_str = mod.__class__.__name__
            mode_type = config.cfg["mod_dict"][mod_type_str]
            mod_info = mod_types[mode_type]
            # build dummy_mod_extra_config for module initialization
            placeholder_tensor = torch.tensor(1.0).to("hpu")
            dummy_mod_scales = ModuleConfig(
                [placeholder_tensor for _ in range(mod_info.num_inputs)],
                [placeholder_tensor for _ in range(mod_info.num_outputs)],
                {name: placeholder_tensor for name in mod_info.param_names},
            )
            if mode_type == "fused_sdpa":
                dummy_mod_scales.inputs.append(placeholder_tensor)  # add amax scale to input scales
            mod_config = scaling_method[mode_type][1](mod, dummy_mod_scales, scale_config)
            dummy_mod_extra_config = ModuleExtraConfig(
                mod_config.inputs,
                mod_config.outputs,
                mod_config.params,
                dummy_mod_scales,
                scale_config,
                )
            # replace bf16 meta weights with FP8 meta weights for loading
            if not config.cfg["fake_quant"] and mod_default_dict[mod_type_str].should_measure_and_quant:
                for param_name in mod_info.param_names:
                    if param_name == "weight":  # only weight is quantized now
                        raw_param = getattr(mod, param_name)
                        param = torch.ones(raw_param.shape, dtype=scale_config["lp_dtype"], device="meta")  # meta tensor
                        delattr(mod, param_name)
                        setattr(mod, param_name, nn.Parameter(param))
            patch_module(mod, dummy_mod_extra_config, mod_default_dict)
            patched_modules.append(name)
            patched_module_types.add(type(mod))
            logger.debug("Patched module name: %s", name)
    logger.debug("Patched module types: %s", patched_module_types)
    logger.debug("Patched modules: %s", patched_modules)
    logger.debug("Total patched modules: %d", len(patched_modules))
    return model


def quantize(model, mod_list):
    """Builds quantization config object that contains for each submodule its quantization functions as preparation for quantization.

    Args:
        model (nn.module): The model that will be quantized.
        mod_list (list, optional): The specific modules that will be quantized in the model.
    """
    config = get_hqt_config(model)
    generate_model_info(model)
    hp_dtype = config.cfg["hp_dtype"]
    lp_dtype = config.cfg["fp8_config"]
    # FIXME make sure this takes unit_scale or measured scale, from Configs
    scaling_method_name = scale_method_mapping[(config.cfg["scale_method"], config.cfg["observer"])]
    scaling_method = scaling_methods[scaling_method_name]
    scale_config = config.cfg["scale_params"]
    scale_config["hp_dtype"] = hp_dtype
    scale_config["lp_dtype"] = lp_dtype
    if config.cfg["mode"] == QuantMode.QUANTIZE:
        measurement = {}
        scale_file = None
        use_stats_files = config.cfg["use_stats_files"]
        if use_stats_files:
            measurement = load_measurements(model, config.cfg["measure_file"])
            scale_file = config.cfg["scale_file"]
        prepare_model(model, mod_list, measurement, scale_file, scaling_method, scale_config)
    elif config.cfg["mode"] == QuantMode.LOAD:
        # no measurement and scale file
        prepare_model_with_dummy_measurement(model, mod_list, scaling_method, scale_config)
    else:
        raise Exception("unexpected mode, expected QuantMode.QUANTIZE or QuantMode.LOAD")
