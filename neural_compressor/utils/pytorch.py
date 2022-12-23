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

"""Pytorch utilities."""

from ..adaptor.pytorch import _cfg_to_qconfig, _cfgs_to_fx_cfgs
from ..adaptor.pytorch import _propagate_qconfig, get_torch_version
from ..adaptor.pytorch import PyTorch_FXAdaptor
from ..adaptor.torch_utils import util
from . import logger
from packaging.version import Version
from torch.quantization import add_observer_, convert
import torch
import torch.quantization as tq
import yaml
import os
import copy

yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                                 lambda loader, node: tuple(loader.construct_sequence(node)))


def _set_sub_module_scale_zeropoint(model, tune_cfg, prefix=''):
    """Set activation scale and zero_point for converted sub modules recursively.

    Args:
        q_model (dir): Int8 model converted from fp32 model. 
                       scale=1, zero_point=0 for each module
        tune_cfg (object): This file provides scale and zero_point of \
                           output activation of each quantized module.
        prefix (string): prefix of op name

    Returns:
        (object): quantized model with scale and zero_point
    """
    for name, module in model.named_children():
        op_name = prefix + '.' + name if prefix != '' else name
        if op_name in tune_cfg['fx_sub_module_list']:
            for key_name in tune_cfg['get_attr'].keys():
                node_name, node_target = key_name.split('--')
                if op_name == node_name:
                    setattr(model, node_target, torch.tensor(tune_cfg['get_attr'][key_name]))
        else:
            _set_sub_module_scale_zeropoint(module, tune_cfg, op_name)


def _set_activation_scale_zeropoint(q_model, tune_cfg):
    """Set activation scale and zero_point for converted model.

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
        if not tune_cfg['fx_sub_module_list']:
            for node_target in tune_cfg['get_attr'].keys():
                setattr(q_model, node_target,
                  torch.tensor(tune_cfg['get_attr'][node_target]))
        else:
            _set_sub_module_scale_zeropoint(q_model, tune_cfg)


def _load_int8_orchestration(model, tune_cfg, stat_dict, example_inputs, **kwargs):
    q_cfgs = torch.quantization.QConfig(
                activation=torch.quantization.FakeQuantize.with_args(
                        dtype=torch.quint8,
                        qscheme=torch.per_tensor_affine,
                        reduce_range=tune_cfg['reduce_range']),
                weight=torch.quantization.default_weight_fake_quant)
    if tune_cfg['framework'] == 'pytorch_fx':
        from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx
        quantized_ops = {op[0]: q_cfgs for op in tune_cfg['quantizable_ops']}
        version = get_torch_version()
        if version < Version("1.11.0-rc1"):
            quantized_ops["default_qconfig"] = None
        else:
            from torch.ao.quantization import default_embedding_qat_qconfig
            for op in tune_cfg['quantizable_ops']:
                if op[1] in ['Embedding', 'EmbeddingBag']:
                    quantized_ops[op[0]] = default_embedding_qat_qconfig
        fx_op_cfgs = _cfgs_to_fx_cfgs(quantized_ops, 'quant_aware_training')
        model.train()
        if tune_cfg['sub_module_list'] is None:
            # pylint: disable=E1123
            if version.release >= Version("1.13.0").release:  # pragma: no cover
                model = prepare_qat_fx(
                    model,
                    fx_op_cfgs,
                    example_inputs=example_inputs,
                    prepare_custom_config=kwargs.get('prepare_custom_config_dict', None)
                    if kwargs is not None else None)
                model = convert_fx(
                    model,
                    convert_custom_config=kwargs.get('convert_custom_config_dict', None)
                    if kwargs is not None else None)
            else:
                model = prepare_qat_fx(
                    model,
                    fx_op_cfgs,
                    prepare_custom_config_dict=kwargs.get('prepare_custom_config_dict', None)
                    if kwargs is not None else None)
                model = convert_fx(
                    model,
                    convert_custom_config_dict=kwargs.get('convert_custom_config_dict', None)
                    if kwargs is not None else None)
        else:
            logger.info('Fx trace of the entire model failed. ' + \
                        'We will conduct auto quantization')
            PyTorch_FXAdaptor.prepare_sub_graph(tune_cfg['sub_module_list'], fx_op_cfgs, \
                                                model, prefix='', is_qat=True)
            PyTorch_FXAdaptor.convert_sub_graph(tune_cfg['sub_module_list'], \
                                                model, prefix='')
    else:
        model.training = True
        model.qconfig = q_cfgs
        torch.quantization.prepare_qat(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
    model.load_state_dict(stat_dict)
    return model


def load(checkpoint_dir=None, model=None, history_cfg=None, **kwargs):
    """Execute the quantize process on the specified model.

    Args:
        checkpoint_dir (dir/file/dict): The folder of checkpoint. 'best_configure.yaml' and 
                                        'best_model_weights.pt' are needed in This directory. 
                                        'checkpoint' dir is under workspace folder and 
                                        workspace folder is define in configure yaml file.
        model (object): fp32 model need to do quantization.
        history_cfg (object): configurations from history.snapshot file.
        **kwargs (dict): contains customer config dict and etc.

    Returns:
        (object): quantized model
    """
    if checkpoint_dir is not None:
        if isinstance(checkpoint_dir, dict):
            stat_dict = checkpoint_dir
        elif os.path.isfile(checkpoint_dir):
            weights_file = checkpoint_dir
            try:
                stat_dict = torch.jit.load(weights_file)
                logger.info("torch.jit.load is used to recovery the int8 model quantized by INC IPEX backend")
            except:
                stat_dict = torch.load(weights_file)
        elif os.path.isdir(checkpoint_dir):
            try:
                weights_file = os.path.join(os.path.abspath(os.path.expanduser(checkpoint_dir)),
                                           'best_model.pt')
                try:
                    stat_dict = torch.jit.load(weights_file)
                    logger.info("torch.jit.load is used to recovery the int8 model quantized by INC IPEX backend")
                except:
                    stat_dict = torch.load(weights_file)
            except:
                tune_cfg_file = os.path.join(os.path.abspath(os.path.expanduser(checkpoint_dir)),
                                            'best_configure.yaml')
                weights_file = os.path.join(os.path.abspath(os.path.expanduser(checkpoint_dir)),
                                            'best_model_weights.pt')
                stat_dict = torch.load(weights_file)
                with open(tune_cfg_file, 'r') as f:
                    tune_cfg = yaml.safe_load(f)
                stat_dict['best_configure'] = tune_cfg
        else:
            logger.error("Unexpected checkpoint type:{}. \
              Only file dir/path or state_dict is acceptable")

        if not isinstance(stat_dict, torch.jit._script.RecursiveScriptModule):
            assert 'best_configure' in stat_dict, \
                    "No best_configure found in the model file, " \
                    "please use the int8 model file generated by INC."
            tune_cfg = stat_dict.pop('best_configure')
    else:
        assert history_cfg is not None, \
          "Need chieckpoint_dir or history_cfg to rebuild int8 model"
        tune_cfg = history_cfg
        stat_dict = None

    version = get_torch_version()
    # Get example inputs for ipex and torch>=1.13.
    example_inputs = None
    if isinstance(stat_dict, torch.jit._script.RecursiveScriptModule) or \
      (version.release >= Version("1.13.0").release and \
      tune_cfg['approach'] != "post_training_dynamic_quant"):
        if "example_inputs" in kwargs:
            example_inputs = kwargs["example_inputs"]
        elif "dataloader" in kwargs:
            from ..adaptor.pytorch import get_example_inputs
            example_inputs = get_example_inputs(model, kwargs["dataloader"])
        else:
            logger.warning("Please provide the example_inputs or a dataloader" +
                            " to get example_inputs for quantized model.")

    if isinstance(stat_dict, torch.jit._script.RecursiveScriptModule):
        q_model = torch.jit.freeze(stat_dict.eval())
        if isinstance(example_inputs, torch.Tensor):
            q_model(example_inputs)
            q_model(example_inputs)
        else:
            q_model(*example_inputs)
            q_model(*example_inputs)
        logger.info("Finish load the model quantized by INC IPEX backend.")
        return q_model

    try:
        q_model = copy.deepcopy(model)
    except Exception as e:                                           # pragma: no cover
        logger.warning("Fail to deep copy the model due to {}, inplace is used now.".
                       format(repr(e)))
        q_model = model

    if 'is_oneshot' in tune_cfg and tune_cfg['is_oneshot']:
        return _load_int8_orchestration(q_model, tune_cfg, stat_dict, example_inputs, **kwargs)

    q_model.eval()
    approach_quant_mode = None
    if tune_cfg['approach'] == "post_training_dynamic_quant":
        approach_quant_mode = 'dynamic'
    elif tune_cfg['approach'] == "post_training_static_quant":
        approach_quant_mode = 'static'

    for _, op_cfg in tune_cfg['op'].items():
        if 'quant_mode' not in op_cfg['activation']:
            op_cfg['activation']['quant_mode'] = approach_quant_mode

    if tune_cfg['approach'] != "post_training_dynamic_quant":
        if version < Version("1.7.0-rc1"):   # pragma: no cover
            q_mapping = tq.default_mappings.DEFAULT_MODULE_MAPPING
        elif version < Version("1.8.0-rc1"):   # pragma: no cover
            q_mapping = \
                tq.quantization_mappings.get_static_quant_module_mappings()
        else:
            q_mapping = \
                tq.quantization_mappings.get_default_static_quant_module_mappings()
    else:
        if version < Version("1.7.0-rc1"):   # pragma: no cover
            q_mapping = \
                tq.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING
        elif version < Version("1.8.0-rc1"):   # pragma: no cover
            q_mapping = \
                tq.quantization_mappings.get_dynamic_quant_module_mappings()
        else:
            q_mapping = \
                tq.quantization_mappings.get_default_dynamic_quant_module_mappings()

    if tune_cfg['framework'] == "pytorch_fx":             # pragma: no cover
        # For torch.fx approach
        assert version >= Version("1.8.0-rc1"), \
                      "Please use PyTroch 1.8 or higher version with pytorch_fx backend"
        from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
        if kwargs is None:
            kwargs = {}
        prepare_custom_config_dict = kwargs.get(
                                        'prepare_custom_config_dict', None)
        convert_custom_config_dict = kwargs.get(
                                        'convert_custom_config_dict', None)

        op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg['approach'])
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg['approach'])
        if not tune_cfg['fx_sub_module_list']:
            tmp_model = q_model
            if tune_cfg['approach'] == "quant_aware_training":
                q_model.train()
                if version > Version("1.12.1"):  # pragma: no cover
                    # pylint: disable=E1123
                    q_model = prepare_qat_fx(q_model,
                                             fx_op_cfgs,
                                             prepare_custom_config=prepare_custom_config_dict,
                                             example_inputs=example_inputs)
                else:
                    q_model = prepare_qat_fx(q_model,
                                             fx_op_cfgs,
                                             prepare_custom_config_dict=prepare_custom_config_dict)
            else:
                if version > Version("1.12.1"):  # pragma: no cover
                    # pylint: disable=E1123
                    q_model = prepare_fx(q_model,
                                         fx_op_cfgs,
                                         prepare_custom_config=prepare_custom_config_dict,
                                         example_inputs=example_inputs)
                else:
                    q_model = prepare_fx(q_model,
                                         fx_op_cfgs,
                                         prepare_custom_config_dict=prepare_custom_config_dict)
            if version > Version("1.12.1"):  # pragma: no cover
                # pylint: disable=E1123
                q_model = convert_fx(q_model,
                  convert_custom_config=convert_custom_config_dict)
            else:
                q_model = convert_fx(q_model,
                  convert_custom_config_dict=convert_custom_config_dict)
            util.append_attr(q_model, tmp_model)
            del tmp_model
        else:
            sub_module_list = tune_cfg['fx_sub_module_list']
            if tune_cfg['approach'] == "quant_aware_training":
                q_model.train()
                PyTorch_FXAdaptor.prepare_sub_graph(sub_module_list,
                                                    fx_op_cfgs,
                                                    q_model,
                                                    prefix='',
                                                    is_qat=True,
                                                    example_inputs=example_inputs)
            else:
                PyTorch_FXAdaptor.prepare_sub_graph(sub_module_list,
                                                    fx_op_cfgs,
                                                    q_model,
                                                    prefix='',
                                                    example_inputs=example_inputs)
            PyTorch_FXAdaptor.convert_sub_graph(sub_module_list, q_model, prefix='')
    else:
        if tune_cfg['approach'] == "post_training_dynamic_quant":
            op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg['approach'])
        else:
            op_cfgs = _cfg_to_qconfig(tune_cfg)

        _propagate_qconfig(q_model, op_cfgs, approach=tune_cfg['approach'])
        # sanity check common API misusage
        if not any(hasattr(m, 'qconfig') and m.qconfig for m in q_model.modules()):
            logger.warn("None of the submodule got qconfig applied. Make sure you "
                        "passed correct configuration through `qconfig_dict` or "
                        "by assigning the `.qconfig` attribute directly on submodules")
        if tune_cfg['approach'] != "post_training_dynamic_quant":
            add_observer_(q_model)
        q_model = convert(q_model, mapping=q_mapping, inplace=True)

    bf16_ops_list = tune_cfg['bf16_ops_list'] if 'bf16_ops_list' in tune_cfg.keys() else []
    if len(bf16_ops_list) > 0 and (version >= Version("1.11.0-rc1")):
        from ..adaptor.torch_utils.bf16_convert import Convert
        q_model = Convert(q_model, tune_cfg)
    if checkpoint_dir is None and history_cfg is not None:
        _set_activation_scale_zeropoint(q_model, history_cfg)
    else:
        q_model.load_state_dict(stat_dict)
    util.get_embedding_contiguous(q_model)
    return q_model
