#
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
"""Block detector for Transformer-based model."""

from ...utils.utility import LazyImport

torch = LazyImport("torch")
from typing import Dict, List, Union

from ...utils import logger
from .util import get_depth, get_dict_at_depth, get_element_under_depth

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
