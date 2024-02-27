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

import copy
import json
import os
import re
import torch
import intel_extension_for_pytorch as ipex
from packaging.version import Version

from neural_compressor.torch.utils import (
    get_ipex_version,
    get_torch_version, 
    logger, 
    simple_inference, 
    unify_op_type_mapping_ipex, 
    TransformerBasedModelBlockPatternDetector,
    get_pattern,
    ipex_config_path,
    get_fuse_ops,
    paser_cfgs,
    get_quantizable_ops_from_cfgs,
)

version = get_torch_version()
ipex_ver = get_ipex_version()


def cfg_to_qconfig(tune_cfg, cfgs, default_cfgs, fuse_ops):  # pragma: no cover
    assert cfgs is not None, "No configure for IPEX int8 model..."
    for key in tune_cfg["op"]:
        try:
            scheme = tune_cfg["op"][key]["activation"]["scheme"]
        except:
            scheme = "asym"
        if scheme not in ["asym", "sym"]:
            scheme = "asym"
        break
    for key in tune_cfg["op"]:
        value = tune_cfg["op"][key]
        pattern = get_pattern(key, fuse_ops)
        assert isinstance(value, dict)
        assert "activation" in value
        if value["activation"]["dtype"] == "fp32":
            if "weight" in value:
                assert value["weight"]["dtype"] == "fp32"
            for op_cfg in cfgs:
                if op_cfg["id"] == key[0]:
                    if key[1] in ["relu_", "add_"]:
                        continue
                    num_inputs = len(op_cfg["inputs_quantized"])
                    num_outputs = len(op_cfg["outputs_quantized"])
                    for i_num in range(num_inputs):
                        op_cfg["inputs_quantized"][i_num] = False
                    for o_num in range(num_outputs):
                        op_cfg["outputs_quantized"][o_num] = False
                    if pattern:
                        if pattern[1] in ["relu_", "add_"]:
                            continue
                        tune_cfg["op"][pattern]["activation"]["dtype"] = "fp32"
                        if "weight" in tune_cfg["op"][pattern]:
                            tune_cfg["op"][pattern]["weight"]["dtype"] = "fp32"
        else:
            for op_cfg in cfgs:
                if op_cfg["id"] == key[0]:
                    if key[1] in ["relu_", "add_"]:
                        continue
                    num_inputs = len(op_cfg["inputs_quantized"])
                    num_outputs = len(op_cfg["outputs_quantized"])
                    for i_num in range(num_inputs):
                        op_cfg["inputs_quantized"][i_num] = default_cfgs[key[0]]["inputs_quantized"][i_num]
                    for o_num in range(num_outputs):
                        op_cfg["outputs_quantized"][o_num] = default_cfgs[key[0]]["outputs_quantized"][o_num]
    with open(ipex_config_path, "w") as write_f:
        json.dump(cfgs, write_f)
    if scheme == "asym":
        return torch.per_tensor_affine
    else:
        return torch.per_tensor_symmetric


def get_quantizable_ops_recursively(model, example_inputs): # pragma: no cover
    """Get all quantizable ops from model.

    Args:
        model (object): input model
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
    Returns:
        quantizable_ops (list): list of tuples of op_name and op_type.
        cfgs (dict): dict of configuration
    """
    quantizable_ops = []
    # group ops by position for transform-based model
    detector = TransformerBasedModelBlockPatternDetector(model)
    detect_result = detector.detect_block()
    attention_block = detect_result.get("attention_blocks", None)
    ffn_blocks = detect_result.get("ffn_blocks", None)
    logger.info(f"Attention Blocks: {len(attention_block)}")
    logger.info(f"FFN Blocks: {len(ffn_blocks)}")
    if not os.path.exists(ipex_config_path):
        assert isinstance(model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"

    if hasattr(model, "save_qconf_summary"):  # pragma: no cover
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
        simple_inference(model, example_inputs, iterations=1)
        model.save_qconf_summary(qconf_summary=ipex_config_path)

    map_op_name_to_fqn = {}
    with open(ipex_config_path, "r") as f:
        cfgs = json.load(f)
        default_cfgs = {}
        fuse_ops = []
        if ipex_ver.release < Version("1.12.0").release:  # pragma: no cover
            default_cfgs = copy.deepcopy(cfgs)
            fuse_ops = get_fuse_ops(cfgs)
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
    return quantizable_ops, cfgs, default_cfgs, fuse_ops
