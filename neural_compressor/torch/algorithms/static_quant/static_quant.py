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

# Copied from neural_compressor/adaptor/torch_utils/gptq.py

import copy
import json
import os
import re

import intel_extension_for_pytorch as ipex
import torch
from packaging.version import Version

from neural_compressor.common.utils import DEFAULT_WORKSPACE
from neural_compressor.torch.utils import get_ipex_version, get_quantizable_ops_from_cfgs, logger, paser_cfgs

from .utility import _dump_model_op_stats, _simple_inference, get_example_inputs

ipex_ver = get_ipex_version()
ipex_config_path = os.path.join(DEFAULT_WORKSPACE, "ipex_config_tmp.json")


def static_config_mapping(configs_mapping):
    tune_cfg = {}
    tune_cfg["op"] = configs_mapping
    for (op_name, op_type), cfg in configs_mapping.items():
        tune_cfg["op"][(op_name, op_type)] = {
            "weight": {
                "dtype": cfg.w_dtype,
                "scheme": "sym",
                "granularity": cfg.w_granularity,
                "algorithm": cfg.w_algo,
            },
            "activation": {
                "dtype": cfg.act_dtype,
                "scheme": "sym" if cfg.act_sym else "asym",
                "granularity": cfg.act_granularity,
                "algorithm": cfg.act_algo,
            },
        }
    return tune_cfg


def static_quantize(model, configs_mapping, run_fn, run_args, example_inputs, inplace):
    tune_cfg = static_config_mapping(configs_mapping)
    assert isinstance(model, torch.nn.Module), "only support torch module"
    q_model = quantize(model, tune_cfg, run_fn, example_inputs, inplace)
    logger.info("Static quantization done.")
    return q_model


def quantize(model, tune_cfg, run_fn, example_inputs, inplace):
    """Execute the quantize process on the specified model.

    Args:
        model (object): model need to do quantization, it is Neural Compressor model.
        tune_cfg (dict): quantization config.
        dataloader (object): calibration dataset.

    Returns:
        (dict): quantized model
    """
    # IPEX bug #1: deepcopied prepared model cannot do calibration, need model
    # q_model is useless, but we need to copy other attributes, and pass the converted
    # model to q_model.
    if inplace:
        q_model = model
    else:
        try:
            q_model = copy.deepcopy(model)
        except Exception as e:  # pragma: no cover
            logger.warning("Fail to deep copy the model due to {}, inplace is used now.".format(repr(e)))
            q_model = model

    # qscheme = _cfg_to_qconfig(tune_cfg)  # Update json file in self.ipex_config_path
    iterations = 1
    model.eval()

    if ipex_ver.release >= Version("1.12.0").release:
        # Check save_qconf_summary part is a workaround for IPEX bug.
        # Sometimes the prepared model from get_op_capablitiy loss this attribute
        if not hasattr(model, "save_qconf_summary") or not hasattr(model, "load_qconf_summary"):
            from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig

            if ipex_ver.release >= Version("2.1").release:
                static_qconfig = ipex.quantization.default_static_qconfig_mapping
            else:
                static_qconfig = QConfig(
                    activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
                )
            if isinstance(example_inputs, dict):
                model = ipex.quantization.prepare(
                    model, static_qconfig, example_kwarg_inputs=example_inputs, inplace=inplace
                )
            else:
                model = ipex.quantization.prepare(model, static_qconfig, example_inputs=example_inputs, inplace=inplace)
        model.load_qconf_summary(qconf_summary=ipex_config_path)
        run_fn(model)
        model.save_qconf_summary(qconf_summary=ipex_config_path)
        q_model = _ipex_post_quant_process(model, q_model, example_inputs, inplace=inplace)
    """
    else:
        # for IPEX version < 1.12
        ipex_conf = ipex.quantization.QuantConf(
            configure_file=ipex_config_path, qscheme=qscheme
        )  # pylint: disable=E1101
        run_fn(model)
        ipex_conf.save(ipex_config_path)
        ipex_conf = ipex.quantization.QuantConf(ipex_config_path)  # pylint: disable=E1101
        q_model = ipex.quantization.convert(
            q_model, ipex_conf, example_inputs, inplace=True
        )  # pylint: disable=E1121
    """
    with open(ipex_config_path, "r") as f:
        q_model.tune_cfg = json.load(f)
    q_model.ipex_config_path = ipex_config_path
    if ipex_ver.release >= Version("1.12.0").release:
        _dump_model_op_stats(tune_cfg)
    return q_model


def _ipex_post_quant_process(model, q_model, example_inputs, inplace=False):
    q_model = ipex.quantization.convert(model, inplace=inplace)
    with torch.no_grad():
        try:
            if isinstance(example_inputs, dict):
                q_model = torch.jit.trace(q_model, example_kwarg_inputs=example_inputs)
            else:
                q_model = torch.jit.trace(q_model, example_inputs)
            q_model = torch.jit.freeze(q_model.eval())
        except:
            if isinstance(example_inputs, dict):
                q_model = torch.jit.trace(q_model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False)
            else:
                q_model = torch.jit.trace(q_model, example_inputs, strict=False)
            q_model = torch.jit.freeze(q_model.eval())
    # After freezing, run 1 time to warm up the profiling graph executor to insert prim::profile
    # At the 2nd run, the llga pass will be triggered and the model is turned into
    # an int8 model: prim::profile will be removed and will have LlgaFusionGroup in the graph
    _simple_inference(q_model, example_inputs, iterations=2)
    return q_model


def _get_quantizable_ops_recursively(model, example_inputs):
    """Get all quantizable ops from model.

    Args:
        model (object): input model
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
    Returns:
        quantizable_ops (list): list of tuples of op_name and op_type.
    """
    quantizable_ops = []
    # group ops by position for transform-based model
    from .utility import TransformerBasedModelBlockPatternDetector

    detector = TransformerBasedModelBlockPatternDetector(model)
    detect_result = detector.detect_block()
    attention_block = detect_result.get("attention_blocks", None)
    ffn_blocks = detect_result.get("ffn_blocks", None)
    logger.info(f"Attention Blocks: {len(attention_block)}")
    logger.info(f"FFN Blocks: {len(ffn_blocks)}")
    if not os.path.exists(ipex_config_path):
        assert isinstance(model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"

    if hasattr(model, "save_qconf_summary"):
        os.makedirs(os.path.dirname(ipex_config_path), exist_ok=True)
        model.save_qconf_summary(qconf_summary=ipex_config_path)
    else:
        model.eval()

        # create a quantization config file for intel pytorch extension model
        os.makedirs(os.path.dirname(ipex_config_path), exist_ok=True)
        assert example_inputs is not None, "IPEX need q_dataloader or example_inputs to prepare the model"
        from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig

        if ipex_ver.release >= Version("2.1").release:
            # HistogramObserver will cause a performance issue.
            # static_qconfig = ipex.quantization.default_static_qconfig_mapping
            qconfig = QConfig(
                activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
            )
            from torch.ao.quantization import QConfigMapping

            static_qconfig = QConfigMapping().set_global(qconfig)
        else:
            static_qconfig = QConfig(
                activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
            )

        if isinstance(example_inputs, dict):
            model = ipex.quantization.prepare(model, static_qconfig, example_kwarg_inputs=example_inputs, inplace=True)
        else:
            model = ipex.quantization.prepare(model, static_qconfig, example_inputs=example_inputs, inplace=True)
        _simple_inference(model, example_inputs, iterations=1)
        model.save_qconf_summary(qconf_summary=ipex_config_path)

    map_op_name_to_fqn = {}
    with open(ipex_config_path, "r") as f:
        cfgs = json.load(f)

        from .utility import unify_op_type_mapping_ipex

        if ipex_ver.release < Version("1.12.0").release:  # pragma: no cover
            for op_cfg in cfgs:
                if op_cfg["name"] in unify_op_type_mapping_ipex:
                    quantizable_ops.append((op_cfg["id"], unify_op_type_mapping_ipex[op_cfg["name"]]))
                else:
                    re_flag = False
                    for pattern, unify_op_type in unify_op_type_mapping_ipex["re"].items():
                        if re.match(pattern, op_cfg["name"]):
                            re_flag = True
                            quantizable_ops.append((op_cfg["id"], unify_op_type))
                            break
                    if not re_flag:
                        quantizable_ops.append((op_cfg["id"], op_cfg["name"]))
        else:
            (
                ops_name,
                op_infos_from_cfgs,
                input_tensor_id_op_name,
                output_tensor_id_op_name,
            ) = paser_cfgs(cfgs)
            quantizable_op_names = get_quantizable_ops_from_cfgs(ops_name, op_infos_from_cfgs, input_tensor_id_op_name)
            for name in quantizable_op_names:
                # name : list
                if len(name) == 1:
                    module_key = name[0][0]
                    op_cfg_id = name[0][2]
                    ipex_op_type = cfgs[module_key]["q_op_infos"][op_cfg_id]["op_type"]
                    module_fqn = cfgs[module_key]["q_op_infos"][op_cfg_id].get("fqn", None)

                    if ipex_op_type in unify_op_type_mapping_ipex:
                        quantizable_ops.append((tuple(name), unify_op_type_mapping_ipex[ipex_op_type]))
                        map_op_name_to_fqn[(tuple(name), ipex_op_type)] = module_fqn
                    else:
                        re_flag = False
                        for pattern, unify_op_type in unify_op_type_mapping_ipex["re"].items():
                            if re.match(pattern, ipex_op_type):
                                re_flag = True
                                quantizable_ops.append((tuple(name), unify_op_type))
                                map_op_name_to_fqn[(tuple(name), unify_op_type)] = module_fqn
                                break
                        if not re_flag:
                            quantizable_ops.append((tuple(name), ipex_op_type))
                            map_op_name_to_fqn[(tuple(name), ipex_op_type)] = module_fqn
                else:
                    op_type = ""
                    for op_name in name:
                        module_key = op_name[0]
                        op_cfg_id = op_name[2]
                        single_op_type = cfgs[module_key]["q_op_infos"][op_cfg_id]["op_type"]
                        if single_op_type in unify_op_type_mapping_ipex:
                            single_op_type = unify_op_type_mapping_ipex[single_op_type]
                        op_type += "&" + single_op_type if op_type else single_op_type
                    quantizable_ops.append((tuple(name), op_type))
                    _module_key = name[0][0]
                    _op_cfg_id = name[0][2]
                    module_fqn = cfgs[_module_key]["q_op_infos"][_op_cfg_id]["fqn"]
                    map_op_name_to_fqn[(tuple(name), op_type)] = module_fqn

    logger.debug("Map op name to fqn: ")
    logger.debug(map_op_name_to_fqn)
    logger.info("Attention Blocks : ")
    logger.info(attention_block)
    logger.info("FFN Blocks : ")
    logger.info(ffn_blocks)
    return quantizable_ops
