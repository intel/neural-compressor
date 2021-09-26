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

from ..adaptor.pytorch import _cfg_to_qconfig, _cfgs_to_fx_cfgs
from ..adaptor.pytorch import _propagate_qconfig, get_torch_version
from ..adaptor.pytorch import PyTorchVersionMode
from . import logger
import torch
from torch.quantization import add_observer_, convert
import torch.quantization as tq
import yaml
import os
import copy

yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                                 lambda loader, node: tuple(loader.construct_sequence(node)))


def set_activation_scale_zeropoint(q_model, tune_cfg):
    """set activation scale and zero_point for converted model.

    Args:
        q_model (dir): Int8 model converted from fp32 model. 
                       scale=1, zero_point=0 for each module
        tune_cfg (object): This file provides scale and zero_point of \
                           output activation of each quantized module.

    Returns:
        (object): quantized model with scale and zero_point
    """
    # pylint: disable=not-callable
    # tune_ops splits tune_cfg['op'].keys() into {op_name: op_type}
    if tune_cfg['approach'] == "post_training_dynamic_quant":
        return
    tune_ops = dict()
    for key in tune_cfg['op']:
        tune_ops[key[0]] = key[1]
    for name, module in q_model.named_modules():
        if name in tune_ops.keys():
            key = (name, tune_ops[name])
            value = tune_cfg['op'][key]
            assert isinstance(value, dict)
            if 'scale' in value['activation'].keys():
                module.scale = torch.tensor(value['activation']['scale'])
            if 'zero_point' in value['activation'].keys():
                module.zero_point = torch.tensor(value['activation']['zero_point'])

    if tune_cfg['framework'] == "pytorch_fx":
        # get scale and zero_point of getattr ops.
        for node_target in tune_cfg['get_attr'].keys():
            setattr(q_model, node_target, torch.tensor(tune_cfg['get_attr'][node_target]))


def load(checkpoint_dir=None, model=None, history_cfg=None, **kwargs):
    """Execute the quantize process on the specified model.

    Args:
        checkpoint_dir (dir): The folder of checkpoint.
                              'best_configure.yaml' and 'best_model_weights.pt' are needed
                              in This directory. 'checkpoint' dir is under workspace folder
                              and workspace folder is define in configure yaml file.
        model (object): fp32 model need to do quantization.

    Returns:
        (object): quantized model
    """
    if checkpoint_dir is not None:
        tune_cfg_file = os.path.join(os.path.abspath(os.path.expanduser(checkpoint_dir)),
                                    'best_configure.yaml')
        weights_file = os.path.join(os.path.abspath(os.path.expanduser(checkpoint_dir)),
                                    'best_model_weights.pt')
        assert os.path.exists(
            tune_cfg_file), "tune configure file %s didn't exist" % tune_cfg_file
        assert os.path.exists(
            weights_file), "weight file %s didn't exist" % weights_file
        with open(tune_cfg_file, 'r') as f:
            tune_cfg = yaml.safe_load(f)
    else:
        assert history_cfg is not None, "Need chieckpoint_dir or history_cfg to rebuild int8 model"
        tune_cfg = history_cfg

    version = get_torch_version()
    if tune_cfg['approach'] != "post_training_dynamic_quant":
        if version < PyTorchVersionMode.PT17.value:   # pragma: no cover
            q_mapping = tq.default_mappings.DEFAULT_MODULE_MAPPING
        elif version < PyTorchVersionMode.PT18.value:   # pragma: no cover
            q_mapping = \
                tq.quantization_mappings.get_static_quant_module_mappings()
        else:
            q_mapping = \
                tq.quantization_mappings.get_default_static_quant_module_mappings()
    else:
        if version < PyTorchVersionMode.PT17.value:   # pragma: no cover
            q_mapping = \
                tq.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING
        elif version < PyTorchVersionMode.PT18.value:   # pragma: no cover
            q_mapping = \
                tq.quantization_mappings.get_dynamic_quant_module_mappings()
        else:
            q_mapping = \
                tq.quantization_mappings.get_default_dynamic_quant_module_mappings()

    if version < PyTorchVersionMode.PT17.value:   # pragma: no cover
        white_list = \
            tq.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING \
            if tune_cfg['approach'] == 'post_training_dynamic_quant' else \
            tq.default_mappings.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST - \
            {torch.nn.LayerNorm, torch.nn.InstanceNorm3d, torch.nn.Embedding}
    elif version < PyTorchVersionMode.PT18.value:   # pragma: no cover
        white_list = \
            tq.quantization_mappings.get_dynamic_quant_module_mappings() \
            if tune_cfg['approach'] == 'post_training_dynamic_quant' else \
            tq.quantization_mappings.get_qconfig_propagation_list() - \
            {torch.nn.LayerNorm, torch.nn.InstanceNorm3d, torch.nn.Embedding}
    else:
        white_list = \
            tq.quantization_mappings.get_default_dynamic_quant_module_mappings() \
            if tune_cfg['approach'] == 'post_training_dynamic_quant' else \
            tq.quantization_mappings.get_default_qconfig_propagation_list() - \
            {torch.nn.LayerNorm, torch.nn.InstanceNorm3d, torch.nn.Embedding}

    if tune_cfg['approach'] == "post_training_dynamic_quant":
        op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg['approach'])
    else:
        op_cfgs = _cfg_to_qconfig(tune_cfg)

    model.eval()
    try:
        q_model = copy.deepcopy(model)
    except Exception as e:                                           # pragma: no cover
        logger.warning("Fail to deep copy the model due to {}, inplace is used now.".
                       format(repr(e)))
        q_model = model

    if tune_cfg['framework'] == "pytorch_fx":             # pragma: no cover
        # For torch.fx approach
        assert version >= PyTorchVersionMode.PT18.value, \
                      "Please use PyTroch 1.8 or higher version with pytorch_fx backend"
        from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg['approach'])
        if tune_cfg['approach'] == "quant_aware_training":
            q_model.train()
            q_model = prepare_qat_fx(q_model, fx_op_cfgs,
              prepare_custom_config_dict=kwargs['prepare_custom_config_dict']
              if kwargs and kwargs.__contains__('prepare_custom_config_dict') else None)
        else:
            q_model = prepare_fx(q_model, fx_op_cfgs,
              prepare_custom_config_dict=kwargs['prepare_custom_config_dict']
              if kwargs and kwargs.__contains__('prepare_custom_config_dict') else None)
        q_model = convert_fx(q_model,
          convert_custom_config_dict=kwargs['convert_custom_config_dict']
          if kwargs and kwargs.__contains__('convert_custom_config_dict') else None)
        if checkpoint_dir is None and history_cfg is not None:
            set_activation_scale_zeropoint(q_model, history_cfg)
        else:
            weights = torch.load(weights_file)
            q_model.load_state_dict(weights)
        return q_model

    _propagate_qconfig(q_model, op_cfgs, white_list=white_list, approach=tune_cfg['approach'])
    # sanity check common API misusage
    if not any(hasattr(m, 'qconfig') and m.qconfig for m in q_model.modules()):
        logger.warn("None of the submodule got qconfig applied. Make sure you "
                    "passed correct configuration through `qconfig_dict` or "
                    "by assigning the `.qconfig` attribute directly on submodules")
    if tune_cfg['approach'] != "post_training_dynamic_quant":
        add_observer_(q_model)
    q_model = convert(q_model, mapping=q_mapping, inplace=True)
    if checkpoint_dir is None and history_cfg is not None:
        set_activation_scale_zeropoint(q_model, history_cfg)
    else:
        weights = torch.load(weights_file)
        q_model.load_state_dict(weights)
    return q_model
