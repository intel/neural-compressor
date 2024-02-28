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

import json
import os
from typing import Callable, Dict, List, Tuple, Union

import prettytable as pt
import torch
from typing_extensions import TypeAlias

from neural_compressor.common.utils import DEFAULT_WORKSPACE, logger

OP_NAME_AND_TYPE_TUPLE_TYPE: TypeAlias = Tuple[str, Union[torch.nn.Module, Callable]]

# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}

# All constants for torch
WHITE_MODULE_LIST = [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]

ipex_config_path = os.path.join(DEFAULT_WORKSPACE, "ipex_config_tmp.json")


WEIGHT_NAME = "quantized_model.pt"
QCONFIG_NAME = "qconfig.json"


def register_algo(name):
    """Decorator function to register algorithms in the algos_mapping dictionary.

    Usage example:
        @register_algo(name=example_algo)
        def example_algo(model: torch.nn.Module, quant_config: RTNConfig) -> torch.nn.Module:
            ...

    Args:
        name (str): The name under which the algorithm function will be registered.

    Returns:
        decorator: The decorator function to be used with algorithm functions.
    """

    def decorator(algo_func):
        algos_mapping[name] = algo_func
        return algo_func

    return decorator


def fetch_module(model, op_name):
    """Get module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.

    Returns:
        module (object).
    """
    module = model
    name_list = op_name.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            logger.warning(f"The {op_name} is not present in the model.")
            return None
    return module


def set_module(model, op_name, new_module):
    """Set module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.
        new_module (object): the input model.

    Returns:
        module (object).
    """
    name_list = op_name.split(".")
    if len(name_list) == 1:
        setattr(model, name_list[-1], new_module)
        return
    else:
        second_last_module = fetch_module(model, ".".join(name_list[:-1]))
        if second_last_module is None:
            logger.warning(f"Setting skipped as the {op_name} is not present in the model.")
            return None
        else:
            setattr(second_last_module, name_list[-1], new_module)


def get_model_info(model: torch.nn.Module, white_module_list: List[Callable]) -> List[Tuple[str, Callable]]:
    module_dict = dict(model.named_modules())
    filter_result = []
    filter_result_set = set()
    for op_name, module in module_dict.items():
        if isinstance(module, tuple(white_module_list)):
            pair = (op_name, type(module))
            if pair not in filter_result_set:
                filter_result_set.add(pair)
                filter_result.append(pair)
    logger.debug(f"Get model info: {filter_result}")
    return filter_result


def get_double_quant_config(double_quant_type):
    from neural_compressor.torch.utils.constants import DOUBLE_QUANT_CONFIGS

    if double_quant_type is None:
        return {}
    assert double_quant_type in DOUBLE_QUANT_CONFIGS, "Supported double quant configs: {}".format(
        list(DOUBLE_QUANT_CONFIGS.keys())
    )
    return DOUBLE_QUANT_CONFIGS[double_quant_type]


def simple_inference(q_model, example_inputs, iterations=1):
    """The function is used for ipex warm-up inference."""
    for _ in range(iterations):
        if isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
            q_model(*example_inputs)
        elif isinstance(example_inputs, dict):
            q_model(**example_inputs)
        else:
            q_model(example_inputs)


def dump_model_op_stats(tune_cfg):
    """This is a function to dump quantizable ops of model to user.

    Args:
        tune_cfg (dict): quantization config
    Returns:
        None
    """
    res = dict()
    for k, v in tune_cfg["op"].items():
        op_type_list = k[-1].split("><")
        op_type = ""
        for op in op_type_list:
            if "class" in op:
                op_type = (
                    op[op.rfind(".") + 1 : op.rfind("'")]
                    if op_type == ""
                    else op_type + "&" + op[op.rfind(".") + 1 : op.rfind("'")]
                )
            elif "method" in op:
                start = op.find("'") + 1
                if start > 1:
                    op_type = (
                        op[start : op.find("'", start)]
                        if op_type == ""
                        else op_type + "&" + op[start : op.find("'", start)]
                    )
                else:
                    start = op.find("method") + 7
                    op_type = (
                        op[start : op.find(" ", start)]
                        if op_type == ""
                        else op_type + "&" + op[start : op.find(" ", start)]
                    )
            else:
                op_type = op if op_type == "" else op_type + "&" + op
        if op_type not in res.keys():
            res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
        if v["weight"]["dtype"] == "int8":
            res[op_type]["INT8"] += 1
        elif v["weight"]["dtype"] == "fp32":
            res[op_type]["FP32"] += 1

    output_data = [
        [op_type, sum(res[op_type].values()), res[op_type]["INT8"], res[op_type]["BF16"], res[op_type]["FP32"]]
        for op_type in res.keys()
    ]

    Statistics(
        output_data, header="Mixed Precision Statistics", field_names=["Op Type", "Total", "INT8", "BF16", "FP32"]
    ).print_stat()


def get_fuse_ops(default_cfgs):  # pragma: no cover
    elt_wise = ["relu", "sigmoid", "gelu"]
    inplace_ops = ["relu_", "add_"]
    op_patterns = []
    num_ops = len(default_cfgs)
    for cur_id in range(num_ops):
        cur_op = default_cfgs[cur_id]["name"]
        if cur_op == "dropout":
            continue
        inputs = default_cfgs[cur_id]["inputs_flow"]
        num_input = len(inputs)
        pre_ops = {}
        for i_num in range(num_input):
            inp = inputs[i_num]
            for pre_id in range(cur_id):
                pre_op = default_cfgs[pre_id]["name"]
                pre_out = default_cfgs[pre_id]["outputs_flow"]
                num_out = len(pre_out)
                for o_num in range(num_out):
                    if pre_out[o_num] == inp:
                        if cur_op in inplace_ops and (pre_op in ["conv2d", "conv3d", "linear"]):
                            op_patterns.append([(pre_id, pre_op), (cur_id, cur_op)])
                        if cur_op in elt_wise and (pre_op in ["conv2d", "conv3d", "linear", "add"]):
                            op_patterns.append([(pre_id, pre_op), (cur_id, cur_op)])
                        if cur_op == "add":
                            pre_ops[i_num] = [pre_id, pre_op]
        if len(pre_ops) > 0:
            for key, value in pre_ops.items():
                if (
                    value[1] in ["conv2d", "conv3d", "linear"]
                    and default_cfgs[cur_id]["inputs_quantized"][key] is False
                ):
                    op_patterns.append([(value[0], value[1]), (cur_id, cur_op)])
    return op_patterns


def get_depth(d) -> int:
    """Query the depth of the dict."""
    if isinstance(d, dict):
        return 1 + max(get_depth(v) for v in d.values())
    return 0


def get_dict_at_depth(d, target_depth, result, depth=0):
    """Get all sub-dicts that are at a specified depth in a nested dict."""
    if depth == target_depth:
        result.append(d)
        return
    elif depth < target_depth and isinstance(d, dict):
        for k, v in d.items():
            get_dict_at_depth(v, target_depth, result, depth=depth + 1)


def get_element_under_depth(d, ops_lst):
    """Get all values in a nested dict."""
    if isinstance(d, dict):
        for k, v in d.items():
            get_element_under_depth(v, ops_lst)
    else:
        ops_lst.append(d)


def paser_cfgs(cfgs):  # pragma: no cover
    """Parse configs.

    Args:
        cfgs (dict): the input configs.


    Returns:
        ops_name (list): list of op names.
        tune_cfg (dict): dictionary of quantization configuration.
        op_infos_from_cfgs (dict): op infos from configs.
        output_tensor_ids_op_name (dict): dictionary of output tensor op names.
    """
    ops_name = []
    layer_output_infos_ids = []
    op_infos_from_cfgs = {}
    # record input_tensor_id and op_name
    # {"0": [(" ", "q_op_infos", "0"), (" ", "q_op_infos", "1")]}
    input_tensor_ids_op_name = {}
    output_tensor_ids_op_name = {}
    for module_key in cfgs.keys():
        for state in cfgs[module_key]:
            if state == "layer_output_infos":
                for index, op_info in enumerate(cfgs[module_key][state]):
                    name = (module_key, state, index)
                    ops_name.append(name)
                    layer_output_infos_ids.append(op_info["id"])
                    op_infos_from_cfgs[name] = op_info
                continue
            for op_cfg_id in cfgs[module_key][state].keys():
                op_info = cfgs[module_key][state][op_cfg_id]
                name = (module_key, state, op_cfg_id)
                if name not in ops_name:
                    ops_name.append(name)
                else:
                    assert False, "Please check IPEX int8 configure json whether have the same name ops"
                op_infos_from_cfgs[name] = op_info
                input_tensors = op_info["input_tensor_infos"]
                for input_tensor in input_tensors:
                    if "id" not in input_tensor.keys():
                        continue
                    else:
                        input_tensor_id = input_tensor["id"]
                    if input_tensor_id not in input_tensor_ids_op_name.keys():
                        input_tensor_ids_op_name[input_tensor_id] = [name]
                    else:
                        input_tensor_ids_op_name[input_tensor_id].append(name)
                output_tensors = op_info["output_tensor_infos"]
                for output_tensor in output_tensors:
                    if "id" not in output_tensor.keys():
                        continue
                    else:
                        output_tensor_id = output_tensor["id"]
                    if output_tensor_id not in output_tensor_ids_op_name.keys():
                        output_tensor_ids_op_name[output_tensor_id] = [name]
                    else:
                        output_tensor_ids_op_name[output_tensor_id].append(name)
    return ops_name, op_infos_from_cfgs, input_tensor_ids_op_name, output_tensor_ids_op_name


def get_quantizable_ops_from_cfgs(ops_name, op_infos_from_cfgs, input_tensor_ids_op_name):  # pragma: no cover
    """Get quantizable ops from configs, combine fused ops as one op.

    Args:
        ops_name (list): list of op names.
        op_infos_from_cfgs (dict): op infos from configs.
        input_tensor_ids_op_name (dict): dictionary of input tensor op names.

    Returns:
        cfgs (dict).
    """
    quantizable_ops = []
    seen_ops = []
    for name in ops_name:
        start = True
        if name in seen_ops:
            continue
        elif name[1] not in ["q_op_infos"]:
            continue
        else:
            # judge fuse ops the first op
            op_info = op_infos_from_cfgs[name]
            output_tensors = op_info["output_tensor_infos"]
            input_tensors = op_info["input_tensor_infos"]
            start = any(
                [
                    input_tensor["inf_dtype"] != "torch.float32"
                    for input_tensor in input_tensors
                    if "inf_dtype" in input_tensor.keys()
                ]
            )
            if not start:
                continue
            # add quantizable ops, include op and fuse ops.
            q_ops, stack = [], [(name, [])]
            while stack:
                cur_name, cur = stack.pop()
                seen_ops.append(cur_name)
                if cur_name[1] not in ["q_op_infos"]:
                    q_ops.append(cur)
                    break
                op_info = op_infos_from_cfgs[cur_name]
                output_tensors = op_info["output_tensor_infos"]
                for output_tensor in output_tensors:
                    if output_tensor["inf_dtype"] == "torch.qint8" or output_tensor["inf_dtype"] == "torch.quint8":
                        q_ops.append(cur + [cur_name])
                        break
                    try:
                        next_op_names = input_tensor_ids_op_name[output_tensor["id"]]
                        for next_op_name in next_op_names:
                            stack.append((next_op_name, cur + [cur_name]))
                    except:
                        next_op_name = None
                    if next_op_name is None:
                        q_ops.append(cur + [cur_name])
            for q_op in q_ops:
                quantizable_ops.append(q_op)
    return quantizable_ops


def get_pattern(fallback_op, fuse_ops):  # pragma: no cover
    for fuse_pattern in fuse_ops:
        if fuse_pattern[0] == fallback_op:
            if fuse_pattern[1] in ["relu_", "add_"]:
                return None
            else:
                return fuse_pattern[1]
    return None


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


class Statistics:
    """The statistics printer."""

    def __init__(self, data, header, field_names, output_handle=logger.info):
        """Init a Statistics object.

        Args:
            data: The statistics data
            header: The table header
            field_names: The field names
            output_handle: The output logging method
        """
        self.field_names = field_names
        self.header = header
        self.data = data
        self.output_handle = output_handle
        self.tb = pt.PrettyTable(min_table_width=40)

    def print_stat(self):
        """Print the statistics."""
        valid_field_names = []
        for index, value in enumerate(self.field_names):
            if index < 2:
                valid_field_names.append(value)
                continue

            if any(i[index] for i in self.data):
                valid_field_names.append(value)
        self.tb.field_names = valid_field_names
        for i in self.data:
            tmp_data = []
            for index, value in enumerate(i):
                if self.field_names[index] in valid_field_names:
                    tmp_data.append(value)
            if any(tmp_data[1:]):
                self.tb.add_row(tmp_data)
        lines = self.tb.get_string().split("\n")
        self.output_handle("|" + self.header.center(len(lines[0]) - 2, "*") + "|")
        for i in lines:
            self.output_handle(i)


class TransformerBasedModelBlockPatternDetector:
    """Detect the attention block and FFN block in transformer-based model."""

    from . import BLOCK_PATTERNS

    def __init__(self, model: torch.nn.Module, pattern_lst: List[List[Union[str, int]]] = BLOCK_PATTERNS) -> None:
        """Init the block detector.

        Args:
            model: the model to be detected.
            pattern_lst: block pattern list.
        """
        self.model = model
        self.pattern_lst = pattern_lst
        self.pos_info = None

    def detect_block(self) -> Dict[str, List[List[str]]]:
        """Traverse the model definition and return the attention blocks and ffn blocks.

        Returns:

            blocks: A dict include the detected attention blocks and ffn blocks.
        """
        # Step 1: Traverse model definition and record the op position
        if not self.pos_info:
            pos_info = {0: {}}
            self.traverse_model(self.model, result=pos_info)
            self.pos_info = pos_info
        # Step 2: Traverse all blocks in different depths and record the blocks that matched the pattern
        detect_result = []
        for pattern in self.pattern_lst:
            _, result = self._search_pattern(pos_info, pattern)
            if result:
                detect_result.append((result, pattern))
        # Step 3: Get the attention blocks and ffn blocks
        blocks = {"attention_blocks": None, "ffn_blocks": None}
        blocks["attention_blocks"], blocks["ffn_blocks"] = self._group_block(detect_result)
        logger.debug(f'FFN BLOCKS: {blocks["ffn_blocks"]}')
        logger.debug(f'Attention BLOCKS: {blocks["attention_blocks"]}')
        return blocks

    @staticmethod
    def traverse_model(model, prefix="", depth=1, result=None, key=0):
        """Traverse the pytorch model according to its hierarchical structure.

        Args:
            model: input model to be traversed.
            prefix: prefix of module. Defaults to "".
            depth: current traverse depth. Defaults to 1.
            result: depth and its included ops. Defaults to {0: {}}.
            key: current root key. Defaults to 0.
        """
        module_lst = list(model.named_children())
        if len(module_lst) == 0:
            # layer name: 'encoder.layer.7.attention.self.query'
            # model repr: Linear(in_features=768, out_features=768, bias=True)
            # class name: 'Linear'
            result[key] = (prefix, model, model.__class__.__name__)
        for i, (name, sub_module) in enumerate(module_lst, 1):
            indent = "    " * depth
            new_name = prefix + "." + name if prefix != "" else name
            model_type = sub_module.__class__.__name__
            logger.debug(f"Depth: [{depth}]" + indent + f"[{model_type}]{ new_name}")
            sub_key = (depth, i, model_type)
            if sub_key not in result[key]:
                result[key][sub_key] = dict()
            TransformerBasedModelBlockPatternDetector.traverse_model(
                sub_module, prefix=new_name, depth=depth + 1, result=result[key], key=sub_key
            )

    @staticmethod
    def _search_pattern(pos_info: Dict, pattern: List[List[Union[str, int]]]) -> List[List[str]]:
        """Search all blocks that matched the pattern.

        Args:
            pos_info: the position information of ops.
            pattern: block pattern.

        Returns:
            The number of matched blocks and the matched blocks.
        """
        max_depth = get_depth(pos_info)
        matched_cnt = 0
        result = []
        for depth in range(max_depth, -1, -1):
            attention_depth = depth
            depth_block_lst = []
            get_dict_at_depth(pos_info, attention_depth, depth_block_lst, 0)
            target_op_types = set(pair[0] for pair in pattern)
            for i, block in enumerate(depth_block_lst):
                sub_block_lst = []
                get_dict_at_depth(block, 1, sub_block_lst, 0)
                block_pattern = []
                block_result = []
                for sub_block in sub_block_lst:
                    ops_lst = []
                    get_element_under_depth(sub_block, ops_lst)
                    filter_ops = [op for op in ops_lst if op[2] in target_op_types]
                    if len(filter_ops) > 0:
                        sub_block_pattern = [filter_ops[0][2], len(filter_ops)]
                        block_pattern.append(sub_block_pattern)
                        ops_name = [op[0] for op in filter_ops]
                        block_result.append(ops_name)
                if block_pattern == pattern:
                    matched_cnt += 1
                    logger.debug(f"[DEPTH] {depth} [BLOCK] {i},  Found block match pattern {pattern}!!")
                    logger.debug(f"[Block keys] {block.keys()}")
                    logger.debug(f"[Block Ops] { [pair[0] for pair in ops_lst if pair[2] in target_op_types]}")
                    result.append(block_result)
        if matched_cnt > 0:
            logger.info(f" Found {matched_cnt} blocks")
        return matched_cnt, result

    @staticmethod
    def _group_block(detect_result):
        """Collect attention and ffn blocks from detect result."""
        import itertools

        ffn_block_lst = []
        attention_block_lst = []
        for block_lst, pattern in detect_result:
            for block in block_lst:
                # Group the first block as attention blocks and
                # the remaining blocks belong to ffn block.
                if block:
                    attention_block_lst.append(block[0])
                    ffn_block = list(itertools.chain(*block[1:]))
                    if ffn_block:
                        ffn_block_lst.append(ffn_block)
        return attention_block_lst, ffn_block_lst
