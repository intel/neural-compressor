"""Prune utils."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

import re
from collections import UserDict

import numpy as np

from neural_compressor.utils import logger
from neural_compressor.utils.utility import DotDict

from ...config import WeightPruningConfig as WeightPruningConf
from ...utils.utility import LazyImport

torch = LazyImport("torch")
nn = LazyImport("torch.nn")
F = LazyImport("torch.nn.functional")
tf = LazyImport("tensorflow")


def get_sparsity_ratio(pruners, model):
    """Calculate sparsity ratio of a module/layer.

    Returns:
        Three floats.
        elementwise_over_matmul_gemm_conv refers to zero elements' ratio in pruning layers.
        elementwise_over_all refers to zero elements' ratio in all layers in the model.
        blockwise_over_matmul_gemm_conv refers to all-zero blocks' ratio in pruning layers.
    """
    pattern_sparsity_cnt = 0
    element_sparsity_cnt = 0
    if hasattr(model, "model"):
        model = model.model
    for pruner in pruners:
        if "MultiheadAttentionPruner" in type(pruner).__name__:
            logger.info("Calculate multihead-attention sparsity")
            mha_total = 0.0
            mha_sparse = 0.0
            for k, v in pruner.head_masks.items():
                mha_total += v.numel()
                mha_sparse += v.numel() - torch.count_nonzero(v)
            logger.info(f"MHA sparsity: {mha_sparse / mha_total}")
            continue
        modules = pruner.modules
        sparsity_ratio = pruner.pattern.get_sparsity_ratio(pruner.masks)
        cnt = 0
        for key in modules.keys():
            cnt += modules[key].weight.numel()
        pattern_sparsity_cnt += int(cnt * sparsity_ratio)
        for key in pruner.masks.keys():
            block_num = 1
            if pruner.pattern.block:
                block_size = pruner.pattern.block_size[key]
                block_num = block_size[0] * block_size[1]
            element_sparsity_cnt += torch.sum(pruner.masks[key] == 0).data.item() * block_num

    linear_conv_cnt = 0
    param_cnt = 0
    for name, module in model.named_modules():
        if type(module).__name__ in ["Linear"] or re.search(r"Conv.d", type(module).__name__) is not None:
            linear_conv_cnt += module.weight.numel()

    for n, param in model.named_parameters():
        param_cnt += param.numel()
    if linear_conv_cnt == 0:
        blockwise_over_matmul_gemm_conv = 0
        elementwise_over_matmul_gemm_conv = 0
    else:
        blockwise_over_matmul_gemm_conv = float(pattern_sparsity_cnt) / linear_conv_cnt
        elementwise_over_matmul_gemm_conv = float(element_sparsity_cnt) / linear_conv_cnt
    if param_cnt == 0:
        elementwise_over_all = 0
    else:
        elementwise_over_all = float(element_sparsity_cnt) / param_cnt

    logger.info(
        f"elementwise_over_matmul_gemm_conv:{elementwise_over_matmul_gemm_conv},"
        f" elementwise_over_all:{elementwise_over_all},"
        f"blockwise_over_matmul_gemm_conv:{blockwise_over_matmul_gemm_conv}"
    )

    return elementwise_over_matmul_gemm_conv, elementwise_over_all, blockwise_over_matmul_gemm_conv


def get_sparsity_ratio_tf(pruners, model):
    """Calculate sparsity ratio of a module/layer.

    Returns:
        Three floats.
        elementwise_over_matmul_gemm_conv refers to zero elements' ratio in pruning layers.
        elementwise_over_all refers to zero elements' ratio in all layers in the model.
        blockwise_over_matmul_gemm_conv refers to all-zero blocks' ratio in pruning layers.
    """
    pattern_sparsity_cnt = 0
    element_sparsity_cnt = 0
    if hasattr(model, "model"):
        model = model.model
    for pruner in pruners:
        modules = pruner.modules
        sparsity_ratio = pruner.pattern.get_sparsity_ratio(pruner.masks)
        cnt = 0
        for key in modules.keys():
            cnt += modules[key].get_weights()[0].size
        pattern_sparsity_cnt += int(cnt * sparsity_ratio)
        for key in pruner.masks.keys():
            block_num = 1
            if pruner.pattern.block:
                block_size = pruner.pattern.block_size[key]
                block_num = block_size[0] * block_size[1]
            element_sparsity_cnt += np.sum(pruner.masks[key] == 0) * block_num

    linear_conv_cnt = 0
    param_cnt = 0
    for layer in model.layers:
        if layer.__class__.__name__ in ["Dense"] or re.search(r"Conv.d", layer.__class__.__name__) is not None:
            linear_conv_cnt += layer.get_weights()[0].size

    for layer in model.layers:
        if bool(layer.weights):
            weights = layer.get_weights()[0]
            param_cnt += weights.size
    if linear_conv_cnt == 0:
        blockwise_over_matmul_gemm_conv = 0
        elementwise_over_matmul_gemm_conv = 0
    else:
        blockwise_over_matmul_gemm_conv = float(pattern_sparsity_cnt) / linear_conv_cnt
        elementwise_over_matmul_gemm_conv = float(element_sparsity_cnt) / linear_conv_cnt
    if param_cnt == 0:
        elementwise_over_all = 0
    else:
        elementwise_over_all = float(element_sparsity_cnt) / param_cnt

    logger.info(
        f"elementwise_over_matmul_gemm_conv:{elementwise_over_matmul_gemm_conv},"
        f" elementwise_over_all:{elementwise_over_all},"
        f"blockwise_over_matmul_gemm_conv:{blockwise_over_matmul_gemm_conv}"
    )

    return elementwise_over_matmul_gemm_conv, elementwise_over_all, blockwise_over_matmul_gemm_conv


def check_config(prune_config):
    """Check if the configuration dict is valid for running Pruning object.

    Args:
        prune_config: A config dict object that contains Pruning parameters and configurations.

    Returns:
        None if everything is correct.

    Raises:
        AssertionError.
    """
    assert prune_config["start_step"] >= 0, "start_step should be greater than 0"
    assert prune_config["end_step"] >= -1, "end_step should be greater than 0"
    assert prune_config["end_step"] >= prune_config["start_step"], "end_step should be greater than start_step"
    assert (
        prune_config["target_sparsity"] >= 0 and prune_config["target_sparsity"] < 1.0
    ), "begin_pruning_step should be in range [0,1)"
    assert prune_config["pruning_frequency"] > 0, "pruning_frequency should be greater than 0"
    assert (
        prune_config["max_sparsity_ratio_per_op"] >= 0 and prune_config["max_sparsity_ratio_per_op"] < 1
    ), "pruning_frequency should be greater than 0"
    assert (
        prune_config["pruning_scope"] == "global" or prune_config["pruning_scope"] == "local"
    ), "only support 'global' and 'local' prune domain"
    try:
        prune_config["resume_from_pruned_checkpoint"] = bool(prune_config["resume_from_pruned_checkpoint"])
    except:
        assert False, "resume_from_pruned_checkpoint should be bool value"
    if "x" in prune_config["pattern"]:
        pattern = prune_config["pattern"].split("_")[-1].split("x")
        if pattern[0] == "channel" or pattern[1] == "channel":
            pass
        else:
            try:
                N = int(pattern[0])
                M = int(pattern[1])
            except:
                assert False, "N or M can't convert to int"
            assert N > 0, "N should be greater than 0"
            assert M > 0, "M should be greater than 0"
    if ":" in prune_config["pattern"]:
        pattern = prune_config["pattern"].split("_")[-1].split(":")
        try:
            N = int(pattern[0])
            M = int(pattern[1])
        except:
            assert False, "N or M can't convert to int"
        assert N > 0, "N should be greater than 0"
        assert M > N, "M should be greater than N"
        max_ratio = float(N) / M
        if prune_config["pruning_type"] != "pattern_lock":
            assert prune_config["target_sparsity"] <= max_ratio, "in N:M pattern, the max sparsity is N/M={}".format(
                max_ratio
            )
        prune_config["max_sparsity_ratio_per_op"] = min(max_ratio, prune_config["max_sparsity_ratio_per_op"])
    if prune_config["reg_coeff"] is not None:
        prune_config["reg_coeff"] = float(prune_config["reg_coeff"])
        assert prune_config["reg_coeff"] >= 0, "only support positive reg_type"
    assert (
        prune_config["min_sparsity_ratio_per_op"] >= 0
        and prune_config["min_sparsity_ratio_per_op"] <= prune_config["max_sparsity_ratio_per_op"]
    ), "min_sparsity_ratio_per_op should in[0, max_sparsity_ratio_per_op]"


def reset_none_to_default(obj, key, default):
    """Set undefined configurations to default values.

    Args:
        obj: A dict{key: value}
        key: A string representing the key in obj.
        default: When the key is not in obj, add key by the default item in original obj.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        if (key not in obj.keys()) or obj[key] is None:
            return default
        else:
            return obj[key]
    else:
        if not hasattr(obj, key) or getattr(obj, key) is None:
            return default
        else:
            return getattr(obj, key)


def update_params(info):
    """Update parameters."""
    if "parameters" in info.keys():
        params = info["parameters"]
        for key in params:
            info[key] = params[key]


def process_weight_config(global_config, local_configs, default_config):
    """Process pruning configurations.

    Args:
        global_config: A config dict object that contains pruning parameters and configurations.
        local_config: A config dict object that contains pruning parameters and configurations.
        default_config: A config dict object that contains pruning parameters and configurations.

    Returns:
        pruners_info: A config dict object that contains pruning parameters and configurations.
    """
    pruners_info = []
    default_all = global_config
    for key in default_config.keys():
        default_all[key] = reset_none_to_default(default_all, key, default_config[key])

    if len(local_configs) == 0:  ##only one
        update_params(default_all)
        check_config(default_all)
        pruner_info = DotDict(default_all)
        pruners_info.append(pruner_info)
    else:  ##TODO need update, in this mode, we ignore the global op names
        for pruner_info in local_configs:
            for key in default_config.keys():
                ##pruner_info[key] = reset_none_to_default(pruner_info, key, global_config[key])
                pruner_info[key] = reset_none_to_default(pruner_info, key, default_all[key])
            update_params(pruner_info)
            check_config(pruner_info)
            pruner_info = DotDict(pruner_info)
            pruners_info.append(pruner_info)

    return pruners_info


def process_yaml_config(global_config, local_configs, default_config):
    """Process the yaml configuration file.

    Args:
        global_config: A config dict object that contains pruning parameters and configurations.
        local_config: A config dict object that contains pruning parameters and configurations.
        default_config: A config dict object that contains pruning parameters and configurations.

    Returns:
        pruners_info: A config dict object that contains pruning parameters and configurations.
    """
    pruners_info = []
    default_all = global_config
    for key in default_config.keys():
        default_all[key] = reset_none_to_default(default_all, key, default_config[key])
    if len(local_configs) == 0:
        update_params(default_all)
        check_config(default_all)
        pruner_info = DotDict(default_all)
        pruners_info.append(pruner_info)

    else:  ##TODO need update, in this mode, we ignore the global op names
        for pruner in local_configs:
            for key in default_config.keys():
                pruner_info = pruner.pruner_config
                pruner_info[key] = reset_none_to_default(pruner_info, key, default_all[key])
            update_params(pruner_info)
            check_config(pruner_info)
            pruner_info = DotDict(pruner_info)
            pruners_info.append(pruner_info)

    return pruners_info


def check_key_validity(template_config, user_config):
    """Check the validity of keys.

    Args:
        template_config: A default config dict object that contains pruning parameters and configurations.
        user_config: A user config dict object that contains pruning parameters and configurations.
    """

    def check_key_validity_dict(template_config, usr_cfg_dict):
        """Check the validity of keys in the dict..

        Args:
            template_config: A default config dict object that contains pruning parameters and configurations.
            usr_cfg_dict: A user config dict object that contains pruning parameters and configurations.
        """
        for user_key, user_value in usr_cfg_dict.items():
            if user_key not in template_config.keys():
                logger.warning(f"{user_key} is not supported for config")

    def check_key_validity_prunerv2(template_config, usr_cfg_dict):
        """Check the validity of keys in the prunerv2.

        Args:
            template_config: A default config dict object that contains pruning parameters and configurations.
            usr_cfg_dict: A user config dict object that contains pruning parameters and configurations.
        """
        for user_key, user_value in usr_cfg_dict.pruner_config.items():
            if user_key not in template_config.keys():
                logger.warning(f"{user_key} is not supported for config")

    # multi pruners
    if isinstance(user_config, list):
        for obj in user_config:
            if isinstance(obj, dict):
                check_key_validity_dict(template_config, obj)

    # single pruner, weightconfig or yaml
    elif isinstance(user_config, dict):
        check_key_validity_dict(template_config, user_config)
    return


def process_and_check_config(val):
    """Process and check configurations.

    Args:
        val: A dict that contains the layer-specific pruning configurations.
    """
    default_global_config = {
        "target_sparsity": 0.9,
        "pruning_type": "snip_momentum",
        "pattern": "4x1",
        "op_names": [],
        "excluded_op_names": [],
        "start_step": 0,
        "end_step": 0,
        "pruning_scope": "global",
        "pruning_frequency": 1,
        "min_sparsity_ratio_per_op": 0.0,
        "max_sparsity_ratio_per_op": 0.98,
        "sparsity_decay_type": "exp",
        "criterion_type": "snip_momentum",
        "pruning_op_types": ["Conv", "Linear"],
        "low_memory_usage": False,
    }
    default_local_config = {
        "resume_from_pruned_checkpoint": False,
        "reg_type": None,
        "criterion_reduce_type": "mean",
        "parameters": {"reg_coeff": 0.0},
    }

    params_default_config = {"reg_coeff": 0.0}

    default_config = {}
    default_config.update(default_global_config)
    default_config.update(default_local_config)
    default_config.update(params_default_config)
    if isinstance(val, WeightPruningConf):
        global_configs = val.weight_compression
        pruning_configs = val.pruning_configs
        check_key_validity(default_config, pruning_configs)
        check_key_validity(default_config, global_configs)
        return process_weight_config(global_configs, pruning_configs, default_config)
    else:
        val = val["pruning"]["approach"]["weight_compression_v2"]
        global_configs = val
        pruning_configs = val["pruners"]
        check_key_validity(default_config, pruning_configs)
        check_key_validity(default_config, global_configs)
        return process_yaml_config(global_configs, pruning_configs, default_config)


def process_config(config):
    """Obtain a config dict object from the config file.

    Args:
        config: A string representing the path to the configuration file.

    Returns:
        A config dict object.
    """
    if isinstance(config, WeightPruningConf):
        return process_and_check_config(config)
    else:
        assert False, f"not supported type {config}"


def parse_last_linear(model):
    """Locate the last linear layers of the model.
    While pruning, the final linear often acts like classifier head, which might cause
    accuracy drop.

    Args:
        model: The model to be pruned.
    """
    from .model_slim.pattern_analyzer import ClassifierHeadSearcher

    searcher = ClassifierHeadSearcher(model)
    layer = searcher.search(return_name=True)
    return layer


def parse_last_linear_tf(model):
    """Locate the last linear layers of the model.
    While pruning, the final linear often acts like classifier head, which might cause
    accuracy drop.

    Args:
        model(tf.keras.Model): The model to be pruned.
    """
    from .model_slim.pattern_analyzer import ClassifierHeadSearcherTF

    searcher = ClassifierHeadSearcherTF(model)
    layer = searcher.search(return_name=True)
    return layer


def parse_to_prune(config, model):
    """Keep target pruned layers.

    Args:
        config: A string representing the path to the configuration file.
        model: The model to be pruned.
    """
    modules = {}
    # additional function: exclude last layer (often a classifier head and not suitable to be pruned)
    classifier_head_name = parse_last_linear(model)
    if classifier_head_name is not None:
        config["excluded_op_names"].append(classifier_head_name)
    # locate target layers
    if config["op_names"] is None or config["op_names"] == []:
        config["op_names"] = [".*"]
    for raw in config["op_names"]:
        try:
            pattern = re.compile(raw)
        except:
            assert False, f"regular expression match does not support {raw}"
        for name, module in filter(lambda t: pattern.search(t[0]), model.named_modules()):
            for layer_type in config["pruning_op_types"]:
                if layer_type in type(module).__name__ and hasattr(module, "weight"):
                    modules[name] = module
                    break
    ##remove not to prune layers
    """Drop non-pruned layers."""
    exclude_names = config["excluded_op_names"]
    patterns = [re.compile(s) for s in exclude_names]
    if len(patterns) <= 0:
        return modules
    new_modules = {}
    for name in modules.keys():
        if any([p.search(name) for p in patterns]):
            continue
        new_modules[name] = modules[name]
    return new_modules


def parse_to_prune_tf(config, model):
    """Keep target pruned layers.

    Args:
        config(string): A string representing the path to the configuration file.
        model(tf.keras.Model): The model to be pruned.
    """
    modules = {}
    # additional function: exclude last layer (often a classifier head and not suitable to be pruned)
    classifier_head_name = parse_last_linear_tf(model)
    if classifier_head_name is not None:
        config["excluded_op_names"].append(classifier_head_name)
    # locate target layers
    if config["op_names"] is None or config["op_names"] == []:
        config["op_names"] = [".*"]

    for layer in model.layers:
        for layer_type in config["pruning_op_types"]:
            if layer_type in layer.__class__.__name__ and bool(layer.weights):
                modules[layer.name] = layer

    ##remove not to prune layers
    """Drop non-pruned layers."""
    exclude_names = config["excluded_op_names"]
    patterns = [re.compile(s) for s in exclude_names]
    if len(patterns) <= 0:
        return modules
    new_modules = {}
    for name in modules.keys():
        if any([p.search(name) for p in patterns]):
            continue
        new_modules[name] = modules[name]
    return new_modules


def get_layers(model):
    """Get each layer's name and its module.

    Args:
        model: The model to be pruned.

    Returns: each layer's name and its modules
    """
    layers = []
    search_flag = False

    def unfoldLayer(module):
        """Unfold each layer.

        Args:
            module: The modules.

        Returns: The ModuleList of model
        """
        nonlocal search_flag
        nonlocal layers
        if search_flag:
            return
        if hasattr(type(module), "__name__") and "ModuleList" in type(module).__name__:
            layers = module
            search_flag = True
        layer_list = list(module.named_children())
        for item in layer_list:
            module = item[1]
            if isinstance(module, torch.nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)
    return layers


def move_input_to_device(input, device="cpu"):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = move_input_to_device(input[inp], device)
    elif isinstance(input, list) or isinstance(input, tuple):
        input_res = []
        for inp in input:
            input_res.append(move_input_to_device(inp, device))
        input = input_res
    return input


def collect_layer_inputs(model, layers, layer_idx, layer_inputs, device="cuda:0"):
    """Getting the forward input of a layer.

    Args:
        model: The model to be pruned.
        layers: Selectable layers of the model.
        layer_idx: The layer index.
        layer_inputs: The dataloader or the output of the previous layer.
        device: Specify the type of device to return.
    Returns: input list.
    """
    inputs = []
    other_input_infos = {}
    positional_inputs = []
    model_dev = model.device

    with torch.no_grad():
        if layer_idx == 0:
            layer = layers[layer_idx]

            def forward(_, hidden_states, *positional_args, **kwargs):
                nonlocal inputs
                nonlocal positional_inputs
                nonlocal other_input_infos
                # TODO solve the problem of batchsize!=1
                inputs.append(move_input_to_device(hidden_states, device))
                if len(positional_inputs) <= 0:
                    positional_inputs = move_input_to_device(positional_args, device)
                for key in kwargs.keys():
                    if key not in other_input_infos.keys():
                        other_input_infos[key] = []
                    other_input_infos[key].append(move_input_to_device(kwargs[key], device))
                raise ValueError

            forward_cache = layers[layer_idx].forward
            from functools import partial

            layer.forward = partial(forward, layer)
            for batch in layer_inputs:
                batch = move_input_to_device(batch, model_dev)
                try:
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        model(batch[0])
                    elif isinstance(batch, dict):
                        model(**batch)
                    else:
                        model(batch)
                except ValueError:
                    pass
            layer.forward = forward_cache
        else:
            prev_layer = layers[layer_idx - 1]

            for batch in layer_inputs:
                prev_output = prev_layer(*batch)
                batch[0] = prev_output[0]
                inputs.append(batch)

    return inputs, positional_inputs, other_input_infos


########################################################
## Utility for integrate DeepSpeed
########################################################
import os

USE_DEEPSPEED = False
FLATTEN_DIM2 = 8


def is_deepspeed_available():  # pragma: no cover
    import importlib
    import importlib.metadata as importlib_metadata

    package_exists = importlib.util.find_spec("deepspeed") is not None

    # Check we're not importing a "deepspeed" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    if package_exists:
        try:
            _ = importlib_metadata.metadata("deepspeed")
            return True
        except importlib_metadata.PackageNotFoundError:
            return False


from packaging.version import Version


def get_deepspeed_version():  # pragma: no cover
    try:
        import deepspeed  # pylint: disable=E0401

        deepspeed_version = deepspeed.__version__.split("+")[0]
    except ValueError as e:  # pragma: no cover
        assert False, "Got an unknown version of torch: {}".format(e)
    version = Version(deepspeed_version)
    return version


def check_deepspeed_version():  # pragma: no cover
    version = get_deepspeed_version()
    assert version >= Version("0.12.4"), f"The minimum version requirement of deepspeed is 0.12.4, but got {version}."


USE_DEEPSPEED = os.environ.get("USE_DEEPSPEED", False)
if USE_DEEPSPEED:  # pragma: no cover
    assert is_deepspeed_available(), "Deepspeed is required: `pip install deepspeed>0.12.4"
    check_deepspeed_version()


def safe_get_shape(param):  # pragma: no cover
    if USE_DEEPSPEED:
        # param.ds_tensor is the partitioned tensor
        return param.ds_tensor.shape
    else:
        return param.shape


def safe_get_data(param):  # pragma: no cover
    if USE_DEEPSPEED:
        from deepspeed.utils import safe_get_local_fp32_param  # pylint: disable=E0401

        return safe_get_local_fp32_param(param)
    else:
        return param.data


def safe_get_grad(param):  # pragma: no cover
    if USE_DEEPSPEED:
        from deepspeed.utils import safe_get_local_grad  # pylint: disable=E0401

        return safe_get_local_grad(param)
    else:
        return param.grad


def safe_set_data(param, new_val):  # pragma: no cover
    if USE_DEEPSPEED:
        from deepspeed.utils import safe_set_local_fp32_param  # pylint: disable=E0401

        safe_set_local_fp32_param(new_val, param)
    else:
        param.data = new_val
