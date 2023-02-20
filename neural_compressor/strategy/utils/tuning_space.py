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

from collections import defaultdict, OrderedDict
import os
import re
from typing import Dict, Tuple
from copy import deepcopy
from enum import IntEnum
from ...utils import logger
from .util import OrderedDefaultDict
from .tuning_structs import OpTuningConfig

from .constant import (
    PRECISION_SET,
    PRECISION_SET_V2_0,
    QUANT_MODE_SET,
    QUNAT_BIT_SET,
    TUNING_ITEMS_LST,
    TUNING_ITEM_SET,
    PostTrainingQuantType,
    
    )

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
        details = ['\t' * depth + f"{self.name},  {self.item_type}"]
        for option in self.options:
            if isinstance(option, int) or isinstance(option, str):
                details.append("\t" * depth + str(option))
            else:
                details.append(option.get_details(depth + 1))
        return "\n".join(details)


class TuningSpace:
    """Not displayed in API Docs.
    
    1) capability -> internal format -> merge -> tuning space (tree)
    2) capability -> merge -> internal format -> tuning space (tree)
    
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
        self.root_item = TuningItem(name='root', options=[], item_type='root')
        self.quant_mode_wise_items = defaultdict(set)  # quant_mode/precision_name: {(op_name, op_type),...}
        self.op_type_wise_items = defaultdict(list)  # op_type: {(op_name, op_type), ...}
        # (op_name, op_type): {quant_mode/precision_name: {'act_dtype': ..., 'weight_dtype': ...}}
        self.framework = framework
        self.ops_dtype = defaultdict(OrderedDict) 
        usr_cfg = conf.usr_cfg if conf else None
        self.op_items = {}

        # New tuning space
        # {(op_name, op_type): {(path): data type}}
        self.ops_data_type = OrderedDefaultDict()
        self.ops_attr = {'activation': set(), 'weight': set()}
        # {(op_name, op_type): {path1, path2, ...}
        self.ops_path_set = defaultdict(set)
        
        self._create_tuning_space(capability, usr_cfg)
        
    def _parse_capability(self, capability: Dict) -> None:
        """Parse the capability and construct the tuning space(a tree).

        Args:
            capability: tThe merged framework capability.
        """
        calib = TuningItem(name='calib_sampling_size',
                           options=capability['calib']['calib_sampling_size'],
                           item_type='calib_sampling_size')
        self.root_item.append(calib)
        def _parse(cap, root, path, op_name_type):
            if isinstance(cap, dict):
                for key, val in cap.items():
                    if isinstance(val, dict):
                        if len(path) > 1 and path[-2] == 'precision':
                            self.ops_path_set[op_name_type].add(tuple(path + [key]))
                        tuning_item = TuningItem(name=key, options=[], item_type=key)
                        root.append(tuning_item)
                        _parse(val, tuning_item, path + [key], op_name_type)
                    elif isinstance(val, list):
                        new_key = ('activation', key) if 'activation' in path else ('weight', key)
                        tuning_item = TuningItem(name=new_key, options=val, item_type='method')
                        self.ops_path_set[op_name_type].add(tuple(path))
                        root.append(tuning_item)
                    else:
                        return

        for op_name_type, op_cap in capability['op'].items():
            op_name, op_type = op_name_type
            op_item = TuningItem(name=op_name_type, options=[], item_type='op')
            self.op_type_wise_items[op_type].append(op_item)
            self.root_item.append(op_item)
            self.op_items[op_name_type] = op_item
            _parse(op_cap, op_item, [], op_name_type)
            print( op_item.name)
            for q_option in op_item.options:
                if q_option and q_option.name == 'precision':
                    acc_item = q_option.get_option_by_name('activation')
                    if acc_item and acc_item.options:
                        for dtype_item in acc_item.options:
                            self.quant_mode_wise_items[dtype_item.name].add(op_item)
                else:
                    self.quant_mode_wise_items[q_option.name].add(op_item)

        logger.info("Constructed tuning space.")
        logger.info(self.root_item.get_details())

    def _create_tuning_item(self, tuning_items: Dict, attr_name: str, quant_mode_item: TuningItem):
        for tuning_item_name, options in tuning_items.items():
            if tuning_item_name not in ['dtype', 'quant_mode']:
                name = (attr_name, tuning_item_name)
                tuning_item = TuningItem(name=name, options=options, item_type=name)
                quant_mode_item.append(tuning_item)

    def _merge_op_cfg(self, op_cap, op_user_cfg, fw_op_cap):
        """Merge the op cfg with user cfg.
        
        dtype: ['int8', 'fp32'] -> ('static', ('int8', 'signed')) and ('precision', ('fp32'))
        dtype: ['fp32'] -> ('precision', ('fp32'))
        step1, For dtype, filter the invalid data type. Override the fwk data type if the valid data type is not empty.
        step2. For tuning item, filter the invalid options, Override the options if the valid option is not empty.
        Skip override if the valid data type or options is empty.
        user_precision_set = {}
        op_user_cfg:
            {
                'weight':{
                    'dtype': ['int8', 'fp32'], 
                    'scheme': ['sym'],
                    'algorithm': ['minmax'],
                    'granularity': ['per_tensor']
                },
                'activation': {
                    'dtype': ['uint8'],
                    'scheme': ['asym'],
                    'algorithm': ['minmax'],
                    'granularity': ['per_tensor']
                    }
            }
        """
        from .util import extract_data_type, reverted_data_type
        fw_op_cap = deepcopy(fw_op_cap)
        for att in ['activation', 'weight']:
            if att in fw_op_cap and op_user_cfg.get(att, None) is not None:
                user_dtype_lst = op_user_cfg[att]['dtype'] if op_user_cfg[att]['dtype'] is not None else []
                # Merge the precision part.
                fwk_att_precision_cap = fw_op_cap['precision'][att]
                fwk_precision_lst = list(fwk_att_precision_cap.keys())
                # The intersection of user cfg and fwk capability.
                valid_precision_set = set(fwk_precision_lst).intersection(set(user_dtype_lst))
                if len(valid_precision_set) != 0:
                    # TODO if dtype is ['int8'], no 'fp32'?
                    for precision_name in fwk_precision_lst:
                        if precision_name not in valid_precision_set:
                            fwk_att_precision_cap.pop(precision_name, None)
                # Merge the quantization part.
                quant_modes_lst = list(fw_op_cap.keys())
                quant_modes_lst.remove('precision')
                for quant_mode in quant_modes_lst:
                    data_type_cap = fw_op_cap[quant_mode][att]
                    data_type_lst = list(data_type_cap.keys())
                    fwk_data_type_lst = []
                    for data_type in data_type_lst:
                        for signed_flag in fw_op_cap[quant_mode][att][data_type].keys():
                            fwk_data_type_lst.append(reverted_data_type(signed_flag, data_type))
                    valid_quant_dtype_lst = set(fwk_data_type_lst).intersection(user_dtype_lst)
                    if len(valid_quant_dtype_lst) != 0:
                        # Filter the valid dtype
                        # TODO if dtype is ['fp32']
                        if len(valid_precision_set) == 0:
                            for precision in fwk_precision_lst:
                                fw_op_cap['precision'][att].pop(precision, None)
                                if len(fw_op_cap['precision'][att]) == 0:
                                    fw_op_cap['precision'].pop(att, None)
                                if len(fw_op_cap['precision']) == 0:
                                    fw_op_cap.pop('precision', None)
                        for dtype in fwk_data_type_lst:
                            if dtype not in valid_quant_dtype_lst:
                                signed_flag, data_type = extract_data_type(dtype)
                                fw_op_cap[quant_mode][att][data_type].pop(signed_flag, None)
                            if len(fw_op_cap[quant_mode][att][data_type]) == 0:
                                fw_op_cap[quant_mode][att].pop(data_type, None)
                            if len(fw_op_cap[quant_mode][att]) == 0:
                                fw_op_cap[quant_mode].pop(att, None)
                    else:
                        # No valid quant data type but have valid precision dtype
                        if len(valid_precision_set) != 0:
                            fw_op_cap.pop(quant_mode, None)
                    # Filter the valid options for tuning item
                    if quant_mode in fw_op_cap:
                        for data_type in fw_op_cap[quant_mode][att]:
                            for signed_flag in fw_op_cap[quant_mode][att][data_type]:
                                fwk_items = fw_op_cap[quant_mode][att][data_type][signed_flag]
                                for item_name, item_options in op_user_cfg[att].items():
                                    if item_name not in ['dtype', 'quant_mode'] and item_options:
                                        options_intersection = set(fwk_items[item_name]).intersection(set(item_options))
                                        if len(options_intersection) > 0:
                                            fwk_items[item_name] = [option for option in fwk_items[item_name] if\
                                                option in options_intersection]
        return fw_op_cap

    def _merge_optype_wise_cfg(self, cap: Dict, optype_wise_usr_cfg: Dict, fw_cap: Dict):
        for op_type, op_user_cfg in optype_wise_usr_cfg.items():
            op_lst = [op_name_type for op_name_type in cap['op'] if op_name_type[1] == op_type]
            for op_name_type in op_lst:
                cap['op'][op_name_type] = self._merge_op_cfg(cap['op'][op_name_type], 
                                                             op_user_cfg,
                                                             fw_cap['op'][op_name_type])

    def _merge_model_wise_cfg(self, cap: Dict, model_wise_usr_cfg: Dict, fw_cap: Dict):
        for op_name_type in cap['op'].keys():
            cap['op'][op_name_type] = self._merge_op_cfg(cap['op'][op_name_type], 
                                                         model_wise_usr_cfg,
                                                         fw_cap['op'][op_name_type])

    def _merge_op_wise_cfg(self, cap: Dict, op_wise_usr_cfg: Dict, fw_cap: Dict):
        op_name_types = {key[0]: key for key in cap['op'].keys()}
        for op_name_pattern, op_user_cfg in op_wise_usr_cfg.items():
            op_name_pattern = re.compile(op_name_pattern)
            for op_name in op_name_types:
                if op_name_pattern.fullmatch(op_name):
                    op_name_type = op_name_types[op_name]
                    logger.debug(f"*** Start to merge user config for op: {op_name_type}")
                    cap['op'][op_name_type] = self._merge_op_cfg(cap['op'][op_name_type], 
                                                                 op_user_cfg,
                                                                 fw_cap['op'][op_name_type])
             
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
        if user_cfg['model_wise'] is not None:
            self._merge_model_wise_cfg(capability, user_cfg['model_wise'], fw_capability)
        if user_cfg['optype_wise'] is not None:
            self._merge_optype_wise_cfg(capability, user_cfg['optype_wise'], fw_capability)
        if user_cfg['op_wise'] is not None:
            self._merge_op_wise_cfg(capability, user_cfg['op_wise'], fw_capability)
            
    def _parse_cap_helper(self, cap):
        """Convert the cpa to internal format.
        
        Parsed result:
        (q/p_type, ((a_bits, a_signed), (w_bits,  w_signed )))
        ('static', (('int8', 'signed'), ('int4', 'unsigned')))
        ('static', (('int8', 'signed'),                     ))
        ('static', ( 'int8'                                  )
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
                            'signed':{ # (op_name, op_type): ('static', ((...), ('int4', 'signed')))
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
                        'fp32':{} #(op_name, op_type): ('precision', ('fp32',)) or ('precision', ('fp32','fp32'))
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
        from .util import OrderedDefaultDict, extract_data_type
        cap = deepcopy(cap)
        parsed_cap = OrderedDict() # {(op_name, op_type): parsed_op_cap}
        for op_name_type, op_cap_lst in cap.items():
            parsed_op_cap = OrderedDefaultDict() # {ptq_type/precision, {}}
            parsed_op_cap['precision'] = OrderedDefaultDict()
            # WA for some op have extra weight dtype.
            has_weight = all(['weight' in op_cap for op_cap in op_cap_lst])
            for op_cap in op_cap_lst:
                if 'activation' in op_cap:
                    self.ops_attr['activation'].add(op_name_type)
                attrs_lst = ['activation', 'weight'] if has_weight else ['activation']
                for att in attrs_lst:
                    # Parse the data info for item that has options.
                    if 'activation' in op_cap and 'quant_mode' in op_cap['activation']:
                        quant_mode = op_cap['activation']['quant_mode']
                        att_dtype = op_cap[att]['dtype'][0]
                        signed_flag, _data_type = extract_data_type(att_dtype)
                        for item_name, item_options in op_cap[att].items():
                            if item_name == 'dtype':
                                # The dtype should be a string, need to align with fwk.yaml.
                                self.ops_data_type[op_name_type][(quant_mode, att, _data_type, signed_flag)] = \
                                    item_options[0] if isinstance(item_options, list) else item_options
                            if item_name not in ['dtype', 'quant_mode']:
                                parsed_op_cap[quant_mode][att][_data_type][signed_flag][item_name] = item_options
                    else:
                        # Parse the data info for item　with unique value.
                        att_dtype = op_cap[att]['dtype']
                        parsed_op_cap['precision'][att][att_dtype] = {'dtype': att_dtype}
                        self.ops_data_type[op_name_type][('precision', att, att_dtype)] = att_dtype

            parsed_cap[op_name_type] = parsed_op_cap
        logger.info(f"Parsed cap ............")
        logger.info(parsed_cap)
        logger.info(f"Data type info...")
        logger.info(self.ops_data_type)
        return parsed_cap
    
    def _create_tuning_space(self, capability, usr_cfg):
        """Create tuning space.
        
        step1. merge the capability with usr_cfg
        step2. create the tuning space
        :param capability:
        :param usr_cfg:
        :return:
        """
        # using new tuning space.
        tmp_cap2 = self._parse_cap_helper(deepcopy(capability['op']))
        capability['op'] = tmp_cap2
        if usr_cfg:
            logger.info(f"#############  Before merged with user cfg")
            logger.info(capability)
            self._merge_with_user_cfg(capability, usr_cfg['quantization'])
            logger.info(f"#############  After Merged with user cfg")
            logger.info(capability)
        self._parse_capability(capability)

    def query_item_option(self, op_name_type, path, method_name, method_val):
        """Query the method value, such as scheme, algorithm.

        Args:
            op_name_type: _description_
            path: _description_
            method_name: _description_
            method_val: _description_

        Returns:
            _description_
        """
        # For static/dynamic/fp32/bf16
        if isinstance(path, str):
            path = ('precision', path) if path in PRECISION_SET_V2_0 else (path, 'int8')
        mode_item = self.get_item_by_path((op_name_type, *path))
        if not mode_item: return None
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
        # TODO handle precision
        # For quant_mode static/dynamic/((static, int8), (dynamic, int4))
        # set the first option as the default if the not support the required quant mode
        full_path = self.get_op_default_path_by_pattern(op_name_type, quant_mode)
        config_args = {}
        has_weight = op_name_type in self.ops_attr['weight']
        config_args['activation_dtype'] = self.ops_data_type[op_name_type].get(full_path['activation'])
        if has_weight:
            config_args['weight_dtype'] = self.ops_data_type[op_name_type].get(full_path['weight'])
        for att in full_path:
            mode_item = self.query_quant_mode_item_by_full_path(op_name_type ,full_path[att])
            if mode_item:
                method_args = {method_item.name: method_item.options[0] for method_item in mode_item.options \
                    if mode_item.name in TUNING_ITEM_SET}
                config_args.update(method_args)

        quant_mode = quant_mode if isinstance(quant_mode, str) else quant_mode[0]
        # set the first option as the default for each tuning item
        op_tuning_config = OpTuningConfig(op_name_type[0],
                                          op_name_type[1],
                                          quant_mode,
                                          self,
                                          kwargs=config_args)
        return op_tuning_config
    
    def get_item_by_path(self, path, default=None):
        """Get the item according to the path."""
        
        logger.info(f"Query item with path {path}")
        item = self.root_item
        for val in path:
            if item is None:
                logger.warning(f"Did not found the item according to the path {path}")
                return default
            item = item.get_option_by_name(val)
        if item is None:
            logger.warning(f"Did not found the item according to the path {path}")
        return item

    def get_default_full_path(self, op_name_type, path):
        """Complete the path.

        Args:
            path: incomplete path. ('precision', 'activation', 'fp32'),
              ('precision', 'activation'), ('static', 'activation', ...)
        """
        # For precision
        if path[0] == 'precision':
            # If the path is ('precision', 'activation', dtype), return it directly.
            if len(path) == 3: return path
            assert len(path) == 2, f"Got the path: {path}, please provide the path include activation or weight."
            att_item = self.get_item_by_path((op_name_type, *path))
            if not att_item or len(att_item.options) == 0: 
                logger.info(f"Could not found item for {op_name_type} with path {path}")
                return None
            dtype = att_item.options[0].name
            return (*path, dtype)
        else:
            # For quantization
            assert len(path) >= 2, f"Got the path: {path}, please provide the path include activation or weight."
            if path[-1] == None: path = path[:-1]
            item = self.get_item_by_path((op_name_type, *path))
            new_path = path
            # For path ('static', 'activation', ...)
            #TODO !! double check to avoid infinite loop
            while item:
                item_options = item.options
                if len(item_options) > 0 and isinstance(item_options[0], TuningItem) and \
                    item_options[0].item_type != 'method':
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

    def query_quant_mode_item(self, op_name_type, path, default_dtype='int8', default_att='activation') -> Tuple[TuningItem, Tuple]:
        """Query the tuning item according to the path for specified.
        
        If the path is incomplete, it will return the default tuning item.
        Args:
            op_name_type: _description_
            path: ('static', 'activation')
            
            For example:
                ('static', (('int8', 'signed'),(...)){
                                'scheme': ['sym'],
                                'algorithm': ['minmax', 'kl'],
                                'granularity': ['per_channel','per_tensor'],
                            }
        Returns:
            (tuning item, complete path),Return the specified tuning item whose options can be unfolded and its path.
        """
        # Backward compatible v2.0
        # For static/dynamic/fp32/bf16
        if isinstance(path, str):
            path = ('precision', path) if path in PRECISION_SET_V2_0 else (path, default_dtype)
        path = (path[0], default_att, path[1])
        new_path = (op_name_type, *path)
        item = self.get_item_by_path(new_path)
        return item
    
    
    def _search_pattern_to_internal(self, pattern):
        """Convert the pattern to internal format."""
        # For (mode, data_type), such as ('static', 'int8')
        _act_pattern, _weight_pattern = (pattern[0], 'activation', pattern[1]), (pattern[0], 'weight', pattern[1])
        return _act_pattern, _weight_pattern
    
    
    def query_items_by_quant_mode(self, quant_mode):
        """Collect all op items that support the specified mode.

        Args:
            quant_mode: dynamic/static/bf16/fp32/fp16

        Returns:
            The op item set that support quant model.
        """
        return self.quant_mode_wise_items.get(quant_mode, set())

    def get_op_default_path_by_pattern(self, op_name_type, pattern):
        """Get the default path by quant mode.

        Args:
            op_name_type: (op_name, op_type)
            pattern: 'static', 'dynamic', ('static', 'int8'), ('precision', 'fp32')
        Returns:
            result(Dict): The default full path of activation and weight if have. 
        """
        internal_pattern = pattern_to_internal(pattern)
        full_path = {'activation': None, 'weight': None}
        full_path['activation'], full_path['weight'] = pattern_to_path(internal_pattern)
        result = {}
        has_weight = op_name_type in self.ops_attr['weight']
        att_lst = ['activation', 'weight'] if has_weight else ['activation']
        for att in att_lst:
            result[att] = self.get_default_full_path(op_name_type, full_path[att])
        return result
        
        
        
def get_op_mode_by_query_order(tuning_space: TuningSpace, query_order):
    """Get the op mode according to the query order."""
    quant_mode_wise_items = OrderedDict() # mode, op_item_lst
    pre_items = set()
    # Collect op items supported the specified mode.
    for quant_mode in query_order:
        items = tuning_space.query_items_by_quant_mode(quant_mode)
        filtered_items = list(filter(lambda item: item not in pre_items, items))
        pre_items = pre_items.union(set(items))
        quant_mode_wise_items[quant_mode] = filtered_items

    def initial_op_quant_mode(items_lst, target_quant_mode, op_item_dtype_dict):
        for item in items_lst:
            op_item_dtype_dict[item.name] = target_quant_mode
    op_item_dtype_dict = OrderedDict()
    for quant_mode, quant_mode_items in quant_mode_wise_items.items():
        initial_op_quant_mode(quant_mode_items, quant_mode, op_item_dtype_dict)
    
    
    print(op_item_dtype_dict)
    return op_item_dtype_dict




def pattern_to_internal(pattern, default_dtype='int8'):
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
        pattern = ('precision', pattern) if pattern in PRECISION_SET_V2_0 else (pattern, (None))
    internal_pattern = (pattern[0], ((pattern[1],), (pattern[1],)))
    logger.debug(f"# Convert pattern: {pattern_bk} into internal pattern {internal_pattern}.")
    return internal_pattern

def pattern_to_path(pattern):
    """Convert pattern to path"""
    act_path = (pattern[0], 'activation', *pattern[1][0])
    weight_path = (pattern[0], 'weight', *pattern[1][1])
    return act_path, weight_path

def quant_mode_from_pattern(internal_pattern):
    """"Get quant mode from internal pattern."""
    if internal_pattern[0] == 'precision':
        return internal_pattern[0][0]
    else:
        return internal_pattern[0]




def initial_tuning_cfg_with_quant_mode(op_name_type, quant_mode, tuning_space: TuningSpace) -> OpTuningConfig:
    """Initial the tuning cfg.

    Args:
        op_name: op name
        op_type: op type
        quant_mode: dynamic/static/fp32/bf16/fp16; (dynamic, int8)/(precision, fp32)
        tuning_space: tuning space.
    
    step1, convert the quant_mode into internal format.    
    step2, complete the path based.
    step3, get the mode item.
    step4, use the first option as value for method.
    step5, create the op tuning config.
    """
    
    internal_pattern = pattern_to_internal(quant_mode)
    full_path = {'activation': None, 'weight': None}
    full_path['activation'], full_path['weight'] = pattern_to_path(internal_pattern)
    logger.debug(f"Convert quant_mode: {quant_mode} into \n \
        \tinternal act_path {full_path['activation']} \n \t weight_path: {full_path['weight']}.")
    has_weight = op_name_type in tuning_space.ops_attr['weight']
    
    config_args = {}
    att_lst = ['activation', 'weight'] if has_weight else ['activation']
    for att in att_lst:
        att_full_path = tuning_space.get_default_full_path(op_name_type, full_path[att])
        config_args[att + '_dtype'] =  tuning_space.ops_data_type[op_name_type].get(att_full_path, None)
        att_mode_item = tuning_space.get_item_by_path((op_name_type, *att_full_path))
        if att_mode_item:
            method_args = {att_mode_item.name: att_mode_item.options[0] for att_mode_item in att_mode_item.options \
                            if att_mode_item.name in TUNING_ITEMS_LST}
            config_args.update(method_args) 
    quant_mode = internal_pattern[0]
    # set the first option as the default for each tuning item
    op_tuning_config = OpTuningConfig(op_name_type[0],
                                      op_name_type[1],
                                      quant_mode,
                                      tuning_space,
                                      kwargs=config_args)
    return op_tuning_config