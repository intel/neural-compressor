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

"""Tuning space."""

import itertools
import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

from ...utils import logger
from .constant import TUNING_ITEMS_LST, WEIGHT_ONLY_TUNING_ITEMS_LST
from .tuning_structs import OpTuningConfig
from .utility import OrderedDefaultDict, preprocess_user_cfg, quant_options


class TuningItem:
    """Not displayed in API Docs."""

    def __init__(self, name, options=[], item_type=None):
        """Init the tuning item.

        Args:
            name: tuning item name.
            options: The options. Defaults to [].
            item_type: The item type. Defaults to None.
        """
        self.name = name
        self._options = options
        self.item_type = item_type

    @property
    def options(self):
        """Return all options.

        Returns:
            All options.
        """
        return self._options

    def get_options_name(self):
        """Return the name list of the options."""
        return [o.name for o in self.options]

    def append(self, option):
        """Append option.

        Args:
            option: The option to add.
        """
        self._options.append(option)

    def remove(self, option):
        """Remove option.

        Args:
            option: The option to remove.
        """
        if option in self._options:
            self._options.remove(option)

    def get_option_by_name(self, option_name):
        """Get the option item by name.

        Args:
            option_name: option name.

        Returns:
            option: the queried option.
        """
        for option in self.options:
            if isinstance(option, TuningItem) and option.name == option_name:
                return option
        return None

    def get_details(self, depth=0):
        """Get the tuning item and its options recursively.

        Args:
            depth: recursion depth. Defaults to 0.

        Returns:
            The tuning item and its options as a string.
        """
        details = ["\t" * depth + f"{self.name},  {self.item_type}"]
        for option in self.options:
            if isinstance(option, int) or isinstance(option, str):
                details.append("\t" * depth + str(option))
            else:
                details.append(option.get_details(depth + 1))
        return "\n".join(details)

    def __repr__(self) -> str:
        """Display the tuning item as string.

        Returns:
            msg: the tuning item as string.
        """
        return self.get_details()


class TuningSpace:
    """Not displayed in API Docs.

    1) capability -> internal format -> merge -> tuning space (tree)
    """

    def __init__(self, capability, conf, framework=None):
        """Init the tuning space.

        Args:
            capability: framework capability.
            conf: user configuration
            framework: framework name. Defaults to None.
        """
        self.capability = capability
        self.conf = conf
        self.root_item = TuningItem(name="root", options=[], item_type="root")
        self.quant_mode_wise_items = defaultdict(list)  # quant_mode/precision_name: {(op_name, op_type),...}
        self.op_type_wise_items = defaultdict(list)  # op_type: {(op_name, op_type), ...}
        self.framework = framework
        self.ops_dtype = defaultdict(OrderedDict)
        self._usr_cfg = self._init_usr_cfg()
        self.op_items = {}
        # {(op_name, op_type): {(path): data type}}
        self.ops_data_type = OrderedDefaultDict()
        self.ops_attr = {"activation": set(), "weight": set()}
        # {(op_name, op_type): {path1, path2, ...}
        self.ops_path_set = defaultdict(set)
        self._create_tuning_space(capability, self._usr_cfg)

    def _init_usr_cfg(self):
        """Init user config."""
        usr_cfg = {"quantization": {}}
        usr_cfg["quantization"]["model_wise"] = None
        usr_cfg["quantization"]["optype_wise"] = self.conf.op_type_dict if self.conf else None
        usr_cfg["quantization"]["op_wise"] = self.conf.op_name_dict if self.conf else None
        return usr_cfg

    def _parse_capability(self, capability: Dict) -> None:
        """Parse the capability and construct the tuning space(a tree).

        Args:
            capability: merged framework capability.
        """
        calib = TuningItem(
            name="calib_sampling_size",
            options=capability["calib"]["calib_sampling_size"],
            item_type="calib_sampling_size",
        )
        self.root_item.append(calib)

        def _parse(cap, root, path, op_name_type):
            if isinstance(cap, dict):
                for key, val in cap.items():
                    if isinstance(val, dict):
                        if len(path) > 1 and path[-2] == "precision":
                            self.ops_path_set[op_name_type].add(tuple(path + [key]))
                        tuning_item = TuningItem(name=key, options=[], item_type=key)
                        root.append(tuning_item)
                        _parse(val, tuning_item, path + [key], op_name_type)
                    elif isinstance(val, list):
                        new_key = ("activation", key) if "activation" in path else ("weight", key)
                        tuning_item = TuningItem(name=new_key, options=val, item_type="method")
                        self.ops_path_set[op_name_type].add(tuple(path))
                        root.append(tuning_item)
                    else:
                        return

        for op_name_type, op_cap in capability["op"].items():
            op_name, op_type = op_name_type
            op_item = TuningItem(name=op_name_type, options=[], item_type="op")
            self.op_type_wise_items[op_type].append(op_item)
            self.root_item.append(op_item)
            self.op_items[op_name_type] = op_item
            _parse(op_cap, op_item, [], op_name_type)
            for q_option in op_item.options:
                if q_option and q_option.name == "precision":
                    acc_item = q_option.get_option_by_name("activation")
                    if acc_item and acc_item.options:
                        for dtype_item in acc_item.options:
                            self.quant_mode_wise_items[dtype_item.name].append(op_item)
                else:
                    self.quant_mode_wise_items[q_option.name].append(op_item)

    def _merge_op_cfg(self, cur_op_cap, op_user_cfg, fw_op_cap):
        """Merge the op cfg with user cfg.

        op_user_cfg:{
            'activation':{
                'dtype': ['fp32']
                },
            'weight':{
                'dtype': ['fp32']
                }
            }

        Step1. merge dtype, get the intersection between fw_op_cap and op_user_cfg.
        Step2. merge method options.

        # if dtype and type intersection with precision set -> only keep the intersection precision
        # and remove the quantization.
        # else(no dtype, or no intersection) -> merge the method

        Args:
            cur_op_cap: current capability.
            op_user_cfg: The user capability.
            fw_op_cap: The fwk capability(baseline).

        Returns:
            Return the merged capability.
        """
        fw_op_cap = deepcopy(fw_op_cap)
        new_op_cap = deepcopy(cur_op_cap)
        op_user_cfg = preprocess_user_cfg(op_user_cfg)
        for att in ["activation", "weight"]:
            if op_user_cfg.get(att, None) is not None:
                user_dtype_lst = op_user_cfg[att]["dtype"] if op_user_cfg[att].get("dtype", None) is not None else []
                # Merge the precision part.
                fwk_att_precision_cap = fw_op_cap["precision"].get(att, {})
                fwk_precision_set = set(fwk_att_precision_cap.keys())
                # The intersection of user cfg and fwk capability.
                valid_precision_set = set(fwk_precision_set).intersection(set(user_dtype_lst))
                if len(valid_precision_set) != 0:
                    new_op_cap = dict(filter(lambda item: item[0] == "precision", new_op_cap.items()))
                    new_op_cap["precision"][att] = dict(
                        filter(lambda item: item[0] in valid_precision_set, fw_op_cap["precision"][att].items())
                    )
                else:
                    # Filter the valid options for tuning item
                    for quant_mode in fw_op_cap:
                        if quant_mode not in new_op_cap:
                            new_op_cap[quant_mode] = deepcopy(fw_op_cap[quant_mode])
                        if quant_mode == "precision":
                            continue
                        for data_type in new_op_cap[quant_mode][att]:
                            for signed_flag in new_op_cap[quant_mode][att][data_type]:
                                cur_items = new_op_cap[quant_mode][att][data_type][signed_flag]
                                fwk_items = fw_op_cap[quant_mode][att][data_type][signed_flag]
                                for method_name, method_options in op_user_cfg[att].items():
                                    skip_list = ["dtype", "quant_mode"]
                                    if data_type == "weight_only":
                                        skip_list = ["quant_mode"]
                                    if method_name not in skip_list and method_options:
                                        # filter the method options
                                        options_intersection = set(fwk_items[method_name]).intersection(
                                            set(method_options)
                                        )
                                        # merge with fwk, if intersection -> use intersection
                                        if len(options_intersection) > 0:
                                            cur_items[method_name] = [
                                                option
                                                for option in fwk_items[method_name]
                                                if option in options_intersection
                                            ]
        return new_op_cap

    def _merge_optype_wise_cfg(self, cap: Dict, optype_wise_usr_cfg: Dict, fw_cap: Dict):
        for op_type, op_user_cfg in optype_wise_usr_cfg.items():
            op_type_pattern = re.compile(op_type)
            op_lst = [op_name_type for op_name_type in cap["op"] if op_type_pattern.fullmatch(op_name_type[1])]
            for op_name_type in op_lst:
                cap["op"][op_name_type] = self._merge_op_cfg(
                    cap["op"][op_name_type], op_user_cfg, fw_cap["op"][op_name_type]
                )

    def _merge_op_wise_cfg(self, cap: Dict, op_wise_usr_cfg: Dict, fw_cap: Dict):
        op_name_types = {key[0]: key for key in cap["op"].keys()}
        for op_name_pattern, op_user_cfg in op_wise_usr_cfg.items():
            if isinstance(op_name_pattern, str):
                op_name_pattern = re.compile(op_name_pattern)
                str_flag = True
            else:
                str_flag = False
            for op_name in op_name_types:
                if str_flag and op_name_pattern.fullmatch(str(op_name)) or op_name_pattern == op_name:
                    op_name_type = op_name_types[op_name]
                    cap["op"][op_name_type] = self._merge_op_cfg(
                        cap["op"][op_name_type], op_user_cfg, fw_cap["op"][op_name_type]
                    )

    def _merge_with_user_cfg(self, capability: Dict, user_cfg: Dict):
        """Merge the capability with user config.

        Merge the capability queried from the adaptor with user config in the order of
        model-wise, optype-wise, and op-wise if needed.
        The optype-wise user config will override the model-wise user config for their
        intersection parts, the same as the op-wise and optype-wise.

        Here is an example:
        capability:{
            ('op1','type1'): {
                'item1': [item1_option1, item1_option2, item1_option3],
                'item2': [item2_option1, item2_option2, item2_option3],
                }
            ('op2','type1'): {
                'item1': [item1_option1, item1_option2, item1_option3],
                'item2': [item2_option1, item2_option2, item2_option3],
                }
            ('op3','type2'): {
                'item1': [item1_option1, item1_option2],
                'item2': [item2_option1, item2_option2],
                }
            ('op4','type2'): {
                'item1': [item1_option1, item1_option2],
                'item2': [item2_option1, item2_option2],
                }
                }

        user_config{
            model-wise:{
                'item1': [item1_option1]
            }
            optype-wise: {
                'type1': {
                    'item1': [item1_option1, item1_option2]
                    }}
            op-wise: {
                ('op3','type2'): {
                    'item2': [item2_option1]
                    }}
            }

        # step1. merged with model-wise
        capability:{
            ('op1','type1'): {
                'item1': [item1_option1],
                'item2': [item2_option1, item2_option2, item2_option3],
                }
            ('op2','type1'): {
                'item1': [item1_option1],
                'item2': [item2_option1, item2_option2, item2_option3],
                }
            ('op3','type2'): {
                'item1': [item1_option1],
                'item2': [item2_option1, item2_option2],
                }
            ('op4','type2'): {
                'item1': [item1_option1],
                'item2': [item2_option1, item2_option2],
                }
                }

        # step2. merged with optype-wise
        capability:{
            ('op1','type1'): {
                'item1': [item1_option1, item1_option2],
                'item2': [item2_option1, item2_option2, item2_option3],
                }
            ('op2','type1'): {
                'item1': [item1_option1, item1_option2],
                'item2': [item2_option1, item2_option2, item2_option3],
                }
            ('op3','type2'): {
                'item1': [item1_option1],
                'item2': [item2_option1, item2_option2],
                }
            ('op4','type2'): {
                'item1': [item1_option1],
                'item2': [item2_option1, item2_option2],
                }
                }

        # step3. merged with op-wise
        capability:{
            ('op1','type1'): {
                'item1': [item1_option1, item1_option2],
                'item2': [item2_option1, item2_option2, item2_option3],
                }
            ('op2','type1'): {
                'item1': [item1_option1, item1_option2],
                'item2': [item2_option1, item2_option2, item2_option3],
                }
            ('op3','type2'): {
                'item1': [item1_option1],
                'item2': [item2_option1],
                }
            ('op4','type2'): {
                'item1': [item1_option1],
                'item2': [item2_option1, item2_option2],
                }
                }
        :param capability:
        :param user_cfg:
        :return:
        """
        fw_capability = deepcopy(capability)
        if user_cfg["optype_wise"] is not None:
            self._merge_optype_wise_cfg(capability, user_cfg["optype_wise"], fw_capability)
        if user_cfg["op_wise"] is not None:
            self._merge_op_wise_cfg(capability, user_cfg["op_wise"], fw_capability)

    def _parse_cap_helper(self, cap):
        """Convert the cpa to internal format.

        Parsed result:
        (op_name, op_type):
            {
                'static':{
                    'act':{
                        'int8':{
                            'signed':{ # (op_name, op_type): ('static', (('int8', 'signed'),(...)))
                                'dtype': 'int8',
                                'scheme': ['sym'],
                                'algorithm': ['minmax', 'kl'],
                                'granularity': ['per_channel','per_tensor'],
                            }
                        }
                        'int4':{
                            ...
                        }
                    },
                    'weight':{
                        'int8':{
                            ...
                        }
                        'int4':{
                            'signed':{
                                'dtype': 'int4'
                                'scheme': ['asym'],
                                ...
                            }
                        }
                    }
                },
                'dynamic':{
                    ...
                }
                'precision':{
                    'act':{
                        'fp32':{}
                        'bf16':{}
                    },
                    'weight':{
                        'fp32':{
                            'dtype': 'fp32,
                        },
                        'bf16':{
                            'dtype': 'fp32',
                            },
                    }

                }
            }
        """
        from .utility import OrderedDefaultDict, extract_data_type

        cap = deepcopy(cap)
        parsed_cap = OrderedDict()  # {(op_name, op_type): parsed_op_cap}
        for op_name_type, op_cap_lst in cap.items():
            parsed_op_cap = OrderedDefaultDict()  # {ptq_type/precision, {}}
            parsed_op_cap["precision"] = OrderedDefaultDict()
            # WA for some op have extra weight dtype.
            has_weight = all(["weight" in op_cap for op_cap in op_cap_lst])
            if has_weight:
                self.ops_attr["weight"].add(op_name_type)
            for op_cap in op_cap_lst:
                if "activation" in op_cap:
                    self.ops_attr["activation"].add(op_name_type)
                attrs_lst = ["activation", "weight"] if has_weight else ["activation"]
                for att in attrs_lst:
                    # Parse the data info for item that has options.
                    if "activation" in op_cap and "quant_mode" in op_cap["activation"]:
                        quant_mode = op_cap["activation"]["quant_mode"]
                        att_dtype = op_cap[att]["dtype"][0]
                        signed_flag, _data_type = extract_data_type(att_dtype)
                        if quant_options.quant_type == 3:
                            _data_type = "weight_only"
                        for item_name, item_options in op_cap[att].items():
                            if item_name == "dtype":
                                # The dtype should be a string, need to align with fwk.yaml.
                                self.ops_data_type[op_name_type][(quant_mode, att, _data_type, signed_flag)] = (
                                    item_options[0] if isinstance(item_options, list) else item_options
                                )
                            if item_name not in ["quant_mode"]:
                                parsed_op_cap[quant_mode][att][_data_type][signed_flag][item_name] = item_options
                    else:
                        # Parse the data info for item　with unique value.
                        att_dtype = op_cap[att]["dtype"]
                        if isinstance(att_dtype, list):
                            att_dtype = att_dtype[0]
                        parsed_op_cap["precision"][att][att_dtype] = {"dtype": att_dtype}
                        self.ops_data_type[op_name_type][("precision", att, att_dtype)] = att_dtype

            parsed_cap[op_name_type] = parsed_op_cap
        return parsed_cap

    def _create_tuning_space(self, capability, usr_cfg):
        """Create tuning space.

        steo1. convert the capability into internal format.
        step2. merge the capability with usr_cfg
        step3. create the tuning space
        :param capability:
        :param usr_cfg:
        :return:
        """
        capability["op"] = self._parse_cap_helper(deepcopy(capability["op"]))
        if usr_cfg:
            self._merge_with_user_cfg(capability, usr_cfg["quantization"])
            logger.debug("***********  After Merged with user cfg ***********")
            logger.debug(capability)
        self._parse_capability(capability)

    def query_item_option(self, op_name_type, path, method_name, method_val):
        """Query the method value, such as scheme, algorithm.

        Args:
            op_name_type: (op_name, op_type)
            path: full path
            method_name: method name
            method_val: method value

        Returns:
            Return the query result if exist.
        """
        mode_item = self.get_item_by_path((op_name_type, *path))
        if not mode_item:
            return None
        method_item = mode_item.get_option_by_name(method_name)
        return method_item is not None and method_val in method_item.options

    def get_default_config(self, op_name_type, quant_mode):
        """Get the default tuning config.

        Args:
            op_name_type: (op_name, op_type)
            quant_mode: quantization mode.

        Returns:
            op_tuning_config: the default config according to the specified quantization mode.
        """
        from .tuning_structs import OpTuningConfig

        # For quant_mode static/dynamic/((static, int8), (dynamic, int4))
        # set the first option as the default if the not support the required quant mode
        full_path = self.get_op_default_path_by_pattern(op_name_type, quant_mode)
        config_args = {}
        has_weight = op_name_type in self.ops_attr["weight"]
        config_args["activation_dtype"] = self.ops_data_type[op_name_type].get(full_path["activation"])
        if has_weight:
            config_args["weight_dtype"] = self.ops_data_type[op_name_type].get(full_path["weight"])
        for att in full_path:
            mode_item = self.query_quant_mode_item_by_full_path(op_name_type, full_path[att])
            if mode_item:
                method_args = {
                    method_item.name: method_item.options[0]
                    for method_item in mode_item.options
                    if method_item.name in TUNING_ITEMS_LST
                }
                config_args.update(method_args)

        quant_mode = quant_mode if isinstance(quant_mode, str) else quant_mode[0]
        # set the first option as the default for each tuning item
        op_tuning_config = OpTuningConfig(op_name_type[0], op_name_type[1], quant_mode, self, kwargs=config_args)
        return op_tuning_config

    def get_item_by_path(self, path, default=None):
        """Get the item according to the path."""
        item = self.root_item
        for val in path:
            if item is None:
                logger.debug(f"Did not found the item according to the path {path}")
                return default
            item = item.get_option_by_name(val)
        if item is None:
            logger.debug(f"Did not found the item according to the path {path}")
        return item

    def get_default_full_path(self, op_name_type, path):
        """Complete the path.

        Args:
            op_name_type: (op_name, op_path)
            path: incomplete path.

        Returns:
            new_path: the complete path.
        """
        # For precision
        if path[0] == "precision":
            # If the path is ('precision', 'activation', dtype), return it directly.
            if len(path) == 3:
                return path
            assert len(path) == 2, f"Got the path: {path}, please provide the path include activation or weight."
            att_item = self.get_item_by_path((op_name_type, *path))
            if not att_item or len(att_item.options) == 0:
                logger.debug(f"Could not found item for {op_name_type} with path {path}")
                return None
            dtype = att_item.options[0].name
            return (*path, dtype)
        else:
            # For quantization
            assert len(path) >= 2, f"Got the path: {path}, please provide the path include activation or weight."
            if path[-1] is None:
                path = path[:-1]
            item = self.get_item_by_path((op_name_type, *path))
            new_path = path
            # For path ('static', 'activation', ...)
            while item:
                item_options = item.options
                if (
                    len(item_options) > 0
                    and isinstance(item_options[0], TuningItem)
                    and item_options[0].item_type != "method"
                ):
                    new_path = new_path + (item_options[0].name,)
                    item = item_options[0]
                else:
                    break
            return new_path

    def query_quant_mode_item_by_full_path(self, op_name_type, path) -> Tuple[TuningItem, Tuple]:
        """Query the mode item by full path."""
        new_path = (op_name_type, *path)
        item = self.get_item_by_path(new_path)
        return item

    def query_items_by_quant_mode(self, quant_mode):
        """Collect all op items that support the specified mode.

        Args:
            quant_mode: dynamic/static/bf16/fp32/fp16

        Returns:
            The op item set that support quant model.
        """
        return self.quant_mode_wise_items.get(quant_mode, [])

    def get_op_default_path_by_pattern(self, op_name_type, pattern):
        """Get the default path by quant mode.

        Args:
            op_name_type: (op_name, op_type)
            pattern: 'static', 'dynamic', ('static', 'int8'), ('precision', 'fp32')

        Returns:
            result(Dict): The default full path of activation and weight if have.
        """
        internal_pattern = pattern_to_internal(pattern)
        full_path = {"activation": None, "weight": None}
        full_path["activation"], full_path["weight"] = pattern_to_path(internal_pattern)
        result = {}
        has_weight = op_name_type in self.ops_attr["weight"]
        att_lst = ["activation", "weight"] if has_weight else ["activation"]
        for att in att_lst:
            result[att] = self.get_default_full_path(op_name_type, full_path[att])
        return result

    def get_op_default_path_by_quant_bits(self, op_name_type, quant_bits):
        """Get the full path according to the target bits.

        Args:
            op_name_type: (op name, op type)
            quant_bits: quantization bits, like int4, int8

        Returns:
            A dict includes the full path.
        """
        quant_modes = ["static", "dynamic"]
        attribute_options = ["activation", "weight"]
        quant_bits = [quant_bits]
        support_attributes = {
            "activation": ("precision", "activation", "fp32"),
            "weight": ("precision", "weight", "fp32"),
        }
        for path in itertools.product(quant_modes, attribute_options, quant_bits):
            if self.query_quant_mode_item_by_full_path(op_name_type, path):
                support_attributes[path[1]] = path
        full_path = {}
        for att in support_attributes:
            full_path[att] = self.get_default_full_path(op_name_type, support_attributes[att])
        return full_path

    def collect_op_by_quant_bits(self, quant_bits: str) -> List[TuningItem]:
        """Collect all OP items that either activation or weight supporting the target bits.

        Args:
            quant_bits: the target quantization bits, like int4, int8.
        """
        quant_modes = ["static", "dynamic"]
        attribute_options = ["activation", "weight"]
        quant_bits = [quant_bits]

        quant_op_items = set(self.query_items_by_quant_mode("static")).union(self.query_items_by_quant_mode("dynamic"))
        op_items = []
        for op in quant_op_items:
            for path in itertools.product(quant_modes, attribute_options, quant_bits):
                if self.query_quant_mode_item_by_full_path(op.name, path):
                    op_items.append(op)
                    break
        return op_items


def pattern_to_internal(pattern, default_dtype="int8"):
    """Convert pattern to internal format.

    'static' -> ('static', (('int8'),('int8')))
    'dynamic' -> ('dynamic', (('int8'),('int8')))
    'fp32' -> ('precision', (('fp32'), ('fp32')))
    'bf16' -> ('precision', (('bf16'), ('bf16')))
    ('static', 'int8') -> ('static', (('int8'),('int8')))
    ('dynamic', 'int8') -> ('dynamic', (('int8'),('int8')))
    ('precision', 'fp32') -> ('precision', (('fp32'), ('fp32')))) # (('fp32'), ('fp32')) or ('fp32', 'fp32')
    #TODO to add the support for mixed data type of weight and activation
    """
    from .constant import PRECISION_SET_V2_0

    pattern_bk = pattern
    if isinstance(pattern, str):
        pattern = ("precision", pattern) if pattern in PRECISION_SET_V2_0 else (pattern, (None))
    internal_pattern = (pattern[0], ((pattern[1],), (pattern[1],)))
    return internal_pattern


def pattern_to_path(pattern):
    """Convert pattern to path."""
    act_path = (pattern[0], "activation", *pattern[1][0])
    weight_path = (pattern[0], "weight", *pattern[1][1])
    return act_path, weight_path


def quant_mode_from_pattern(internal_pattern):
    """Get quant mode from internal pattern."""
    if internal_pattern[0] == "precision":
        return internal_pattern[1][0]
    else:
        return internal_pattern[0]


def initial_tuning_cfg_with_quant_mode(op_name_type, quant_mode, tuning_space: TuningSpace) -> OpTuningConfig:
    """Initialize the tuning cfg.

    Args:
        op_name_type: (op name, op type)
        quant_mode: dynamic/static/fp32/bf16/fp16
        tuning_space: tuning space.

    step1, convert the quant_mode into internal format.
    step2, complete the path based.
    step3, get the mode item.
    step4, use the first option as value for method.
    step5, create the op tuning config.

    Returns:
        The initial tuning config.
    """
    internal_pattern = pattern_to_internal(quant_mode)
    full_path = {"activation": None, "weight": None}
    full_path["activation"], full_path["weight"] = pattern_to_path(internal_pattern)
    has_weight = op_name_type in tuning_space.ops_attr["weight"]

    config_args = {}
    att_lst = ["activation", "weight"] if has_weight else ["activation"]
    for att in att_lst:
        att_full_path = tuning_space.get_default_full_path(op_name_type, full_path[att])
        config_args[att + "_dtype"] = tuning_space.ops_data_type[op_name_type].get(att_full_path, None)
        mode_item = tuning_space.get_item_by_path((op_name_type, *att_full_path))
        if mode_item:
            item_list = WEIGHT_ONLY_TUNING_ITEMS_LST if att_full_path[0] == "weight_only" else TUNING_ITEMS_LST
            method_args = {
                method_item.name: method_item.options[0]
                for method_item in mode_item.options
                if method_item.name in item_list
            }
            config_args.update(method_args)
    quant_mode = internal_pattern[0]
    # set the first option as the default for each tuning item
    op_tuning_config = OpTuningConfig(op_name_type[0], op_name_type[1], quant_mode, tuning_space, kwargs=config_args)
    return op_tuning_config
