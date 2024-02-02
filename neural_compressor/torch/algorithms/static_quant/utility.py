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

from collections import UserDict
from typing import Dict, List, Union

import intel_extension_for_pytorch as ipex
import prettytable as pt
import torch
from packaging.version import Version

from neural_compressor.torch.utils import (
    get_depth,
    get_dict_at_depth,
    get_element_under_depth,
    get_torch_version,
    logger,
)

version = get_torch_version()

unify_op_type_mapping_ipex = {
    "Convolution_Relu": "Conv2d",
    "Convolution_Sum_Relu": "Conv2d",
    "Convolution_BatchNorm": "Conv2d",
    "<class 'torch.nn.modules.conv.Conv1d'>": "Conv1d",
    "<class 'torch.nn.modules.conv.Conv2d'>": "Conv2d",
    "<class 'torch.nn.modules.conv.Conv3d'>": "Conv3d",
    "<class 'torch.nn.modules.activation.ReLU'>": "ReLU",
    "<method 'add' of 'torch._C._TensorBase' objects>": "add",  # for IPEX < 2.2
    "<method 'add' of 'torch._C.TensorBase' objects>": "add",  # for IPEX >= 2.2
    "<class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>": "AdaptiveAvgPool2d",
    "Linear_Relu": "Linear",
    "<class 'torch.nn.modules.linear.Linear'>": "Linear",
    "<class 'torch.nn.modules.pooling.MaxPool2d'>": "MaxPool2d",
    "re": {"<built-in method matmul of type object at": "matmul"},
}


BLOCK_PATTERNS = [
    # [['OP_TYPE1', NUM_OPS], ['OP_TYPE2', NUM_OPS], ...]
    [["Linear", 4], ["Linear", 4]],  # TODO add model name
    [["Linear", 2], ["Linear", 2]],  # TODO add model name
    [["Conv1D", 2], ["Conv1D", 2]],  # GPT-2
    [["Linear", 4], ["Linear", 3]],  # Llama
    [["Linear", 4], ["Linear", 2]],  # T5-Encoder, OPT
    [["Linear", 4], ["Linear", 1], ["Linear", 1]],  # Bert
    [["Linear", 4], ["Linear", 4], ["Linear", 2]],  # T5-Decoder
]


def move_input_device(input, device="cpu"):
    """Auto mapping input to device for all kinds of format.

    Args:
        input (torch.tensor): input data
        device (str, optional): target device. Defaults to "cpu".

    Returns:
        input (torch.tensor): input data on target device
    """
    if device == "cpu":
        return input
    if isinstance(input, dict) or isinstance(input, UserDict):
        tmp_input = {}
        for k, inp in input.items():
            tmp_input[k] = move_input_device(inp, device)
        input = tmp_input
    elif isinstance(input, list) or isinstance(input, tuple):
        tmp_input = []
        for inp in input:
            tmp_input.append(move_input_device(inp, device))
        input = tmp_input
    elif isinstance(input, torch.Tensor):
        input = input.to(device)  # pylint: disable=no-member
    return input


def forward_wrapper(model, input):
    """Model forward with device auto mapping.

    Args:
        model (torch.nn.Module): input model
        input (torch.tensor): input data

    Returns:
        output: output data
    """
    try:
        device = next(model.parameters()).device
    except:
        # for RecursiveScriptModule
        device = "cpu"
    input = move_input_device(input, device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        output = model(**input)
    elif isinstance(input, list) or isinstance(input, tuple):
        try:
            output = model(*input)
        except:
            output = model(input)
    else:
        output = model(input)
    return output


def pytorch_forward_wrapper(
    model,
    input,
    conf=None,
    backend="default",
    running_mode="inference",
):
    if (
        version.release < Version("1.12.0").release and backend == "ipex" and running_mode == "calibration"
    ):  # pragma: no cover
        with ipex.quantization.calibrate(conf, default_recipe=True):  # pylint: disable=E1101
            output = forward_wrapper(model, input)
    else:
        output = forward_wrapper(model, input)
    return output


def get_example_inputs(model, dataloader):
    # Suggest set dataloader like calib_dataloader
    if dataloader is None:
        return None
    device = next(model.parameters()).device
    try:
        for idx, (input, label) in enumerate(dataloader):
            input = move_input_device(input, device)
            output = pytorch_forward_wrapper(model, input)
            if isinstance(input, (dict, UserDict)):  # pragma: no cover
                assert version.release >= Version("1.12.0").release, "INC support IPEX version >= 1.12.0"
                if "label" in input.keys():
                    input.pop("label")
                if version.release <= Version("2.0.1").release:
                    return tuple(input.values())
                else:
                    return dict(input)

            if isinstance(input, (list, tuple)):
                return tuple(input)
            if isinstance(input, torch.Tensor):
                return input
            break
    except Exception as e:  # pragma: no cover
        for idx, input in enumerate(dataloader):
            input = move_input_device(input, device)
            output = pytorch_forward_wrapper(model, input)
            if isinstance(input, (dict, UserDict)):  # pragma: no cover
                assert version.release >= Version("1.12.0").release, "INC support IPEX version >= 1.12.0"
                if "label" in input.keys():
                    input.pop("label")
                if version.release <= Version("2.0.1").release:
                    return tuple(input.values())
                else:
                    return dict(input)
            if isinstance(input, list) or isinstance(input, tuple):
                return tuple(input)
            if isinstance(input, torch.Tensor):
                return input
            break
    if idx == 0:
        assert False, "Please checkout the example_inputs format."


def get_pattern(self, fallback_op, fuse_ops):  # pragma: no cover
    for fuse_pattern in fuse_ops:
        if fuse_pattern[0] == fallback_op:
            if fuse_pattern[1] in ["relu_", "add_"]:
                return None
            else:
                return fuse_pattern[1]
    return None


def _simple_inference(q_model, example_inputs, iterations=1):
    """The function is used for ipex warm-up inference."""
    for _ in range(iterations):
        if isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
            q_model(*example_inputs)
        elif isinstance(example_inputs, dict):
            q_model(**example_inputs)
        else:
            q_model(example_inputs)


'''
def _cfg_to_qconfig(tune_cfg, smooth_quant=False):
        """Convert tune configure to quantization config for each op.

        Args:
            tune_cfg (dict): dictionary of tune configure for each op
            ipex_config_path: configure file of Intel PyTorch Extension
        """
        if version.release < Version("1.12.0").release:  # pragma: no cover
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
                pattern = get_pattern(key, self.fuse_ops)
                assert isinstance(value, dict)
                assert "activation" in value
                if value["activation"]["dtype"] == "fp32":
                    if "weight" in value:
                        assert value["weight"]["dtype"] == "fp32"
                    for op_cfg in self.cfgs:
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
                    for op_cfg in self.cfgs:
                        if op_cfg["id"] == key[0]:
                            if key[1] in ["relu_", "add_"]:
                                continue
                            num_inputs = len(op_cfg["inputs_quantized"])
                            num_outputs = len(op_cfg["outputs_quantized"])
                            for i_num in range(num_inputs):
                                op_cfg["inputs_quantized"][i_num] = self.default_cfgs[key[0]]["inputs_quantized"][i_num]
                            for o_num in range(num_outputs):
                                op_cfg["outputs_quantized"][o_num] = self.default_cfgs[key[0]]["outputs_quantized"][
                                    o_num
                                ]
            with open(self.ipex_config_path, "w") as write_f:
                json.dump(self.cfgs, write_f)
            if scheme == "asym":
                return torch.per_tensor_affine
            else:
                return torch.per_tensor_symmetric
        else:
            op_infos = copy.deepcopy(self.op_infos_from_cfgs)
            self.cfgs = torch_utils.util.check_cfg_and_qconfig(
                tune_cfg["op"], self.cfgs, op_infos, self.output_tensor_id_op_name, smooth_quant
            )

            with open(self.ipex_config_path, "w") as write_f:
                json.dump(self.cfgs, write_f, indent=4)
            return None
'''


def _dump_model_op_stats(tune_cfg):
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


class TransformerBasedModelBlockPatternDetector:
    """Detect the attention block and FFN block in transformer-based model."""

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
