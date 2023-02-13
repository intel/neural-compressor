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

from ast import Or
from collections import defaultdict, OrderedDict
import re
from typing import Dict
from copy import deepcopy
from enum import IntEnum
from ...utils import logger

PRECISION_SET = {'bf16', 'fp32'}
QUANT_MODE_SET = {'static', 'dynamic'}
QUNAT_BIT_SET = {'int8', 'uint8', 'int4', 'uint4'}

TUNING_ITEMS_LST = [('activation','scheme'), ('activation','algorithm'), ('activation','granularity'),
                    ('weight','scheme'), ('weight','algorithm'), ('weight','granularity'),
                    'sampling_size']

class PostTrainingQuantType(IntEnum):
    """Post training quantization type."""
    STATIC = 0
    DYNAMIC = 1
    WEIGHT_ONLY = 2

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
        self.quant_mode_wise_items = defaultdict(list)  # quant_mode/precision_name: {(op_name, op_type),...}
        self.op_type_wise_items = defaultdict(list)  # op_type: {(op_name, op_type), ...}
        # (op_name, op_type): {quant_mode/precision_name: {'act_dtype': ..., 'weight_dtype': ...}}
        self.framework = framework
        self.ops_dtype = defaultdict(OrderedDict) 
        usr_cfg = conf.usr_cfg if conf else None
        self.op_items = {}
        self._create_tuning_space(capability, usr_cfg)
        
    def _parse_capability_v2(self, capability:Dict) -> None:
        """Parse the capability and construct the tuning space(a tree)

        Args:
            capability: tThe merged framework capability.
        """
        pass
    

    def _parse_capability(self, capability):
        """Parse the capability and construct the tuning space(a tree)."""
        calib = TuningItem(name='calib_sampling_size',
                           options=capability['calib']['calib_sampling_size'],
                           item_type='calib_sampling_size')
        self.root_item.append(calib)

        for op_name_type, op_cap in capability['op'].items():
            op_name, op_type = op_name_type
            op_item = TuningItem(name=op_name_type, options=[], item_type='op')
            self.op_type_wise_items[op_type].append(op_item)
            self.root_item.append(op_item)
            self.op_items[op_name_type] = op_item
            op_weight_flag = op_cap['op_weight_flag']
            # for other precision capability
            for quant_mode in op_cap['precision']:
                self.quant_mode_wise_items[quant_mode].append(op_item)
                quant_mode_item = TuningItem(name=quant_mode, options=[], item_type='quant_mode')
                op_item.append(quant_mode_item)
                self.ops_dtype[op_name_type][quant_mode] = {'act_dtype': quant_mode}
                if op_weight_flag:
                    self.ops_dtype[op_name_type][quant_mode]['weight_dtype'] = quant_mode
            for quant_mode_flag, quant_cap in op_cap['quant'].items():
                quant_mode_item = TuningItem(name=quant_mode_flag, options=[], item_type='quant_mode')
                op_item.append(quant_mode_item)
                self.quant_mode_wise_items[quant_mode_flag].append(op_item)
                act_dtype = quant_cap['activation']['dtype']
                act_dtype = act_dtype[0] if isinstance(act_dtype, list) else act_dtype
                self.ops_dtype[op_name_type][quant_mode_flag] = {'act_dtype': act_dtype}
                self._create_tuning_item(quant_cap['activation'], 'activation', quant_mode_item)
                if op_weight_flag:
                    self._create_tuning_item(quant_cap['weight'], 'weight', quant_mode_item)
                    weight_dtype = quant_cap['weight']['dtype']
                    weight_dtype = weight_dtype[0] if isinstance(weight_dtype, list) else weight_dtype
                    self.ops_dtype[op_name_type][quant_mode_flag]['weight_dtype'] = weight_dtype
                                                                                       

    def _create_tuning_item(self, tuning_items: Dict, attr_name: str, quant_mode_item: TuningItem):
        for tuning_item_name, options in tuning_items.items():
            if tuning_item_name not in ['dtype', 'quant_mode']:
                name = (attr_name, tuning_item_name)
                tuning_item = TuningItem(name=name, options=options, item_type=name)
                quant_mode_item.append(tuning_item)

    def _merge_op_cfg(self, op_cap, op_user_cfg, fw_op_cap):
        """
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
            if att in op_user_cfg and op_user_cfg[att] is not None:
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
                        if len(valid_precision_set) != 0: # no valid quant data type but have valid precision dtype
                            fw_op_cap.pop(quant_mode, None)
                    # Filter the valid options for tuning item
                    if quant_mode in fw_op_cap:
                        for data_type in fw_op_cap[quant_mode][att]:
                            for signed_flag in fw_op_cap[quant_mode][att][data_type]:
                                fwk_items = fw_op_cap[quant_mode][att][data_type][signed_flag]
                                for item_name, item_options in op_user_cfg[att].items():
                                    if item_name not in ['dtype', 'quant_mode']:
                                        options_intersection = set(fwk_items[item_name]).intersection(set(item_options))
                                        if len(options_intersection) > 0:
                                            fwk_items[item_name] = [option for option in fwk_items[item_name] if\
                                                option in options_intersection]
        return fw_op_cap

    def _merge_op_cfg_v2(self, op_cap, op_user_cfg, fw_op_cap):
        """Merge the framework capability with user config.

        # for precision, merge the options of the tuning items 
        # for quanzation, merge the dtype
        supported_precision_set = {'fp32', 'bf16'}
        user_precision_set = {}
        op_user_cfg:
            {
                'weight':{
                    'dtype': ['int8'],
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
            
        Returns:
            op_cap: merged op capability.

        """
        for key in ['activation', 'weight']:
            if key in op_user_cfg and op_user_cfg[key] is not None:
                user_dtype_lst = op_user_cfg[key]['dtype'] if op_user_cfg[key]['dtype'] is not None else []
                precision_flag = len(PRECISION_SET.intersection(set(user_dtype_lst))) != 0
                quant_flag = len(QUNAT_BIT_SET.intersection(set(user_dtype_lst))) != 0
                if precision_flag and not quant_flag:
                    merged_options = [option for option in user_dtype_lst if option in fw_op_cap['precision']]
                    if not merged_options: 
                        merged_options = fw_op_cap['precision']
                    op_cap['precision'] = merged_options
                    op_cap['quant'] = OrderedDict() # do not do quantization
                    break
                for quant_mode_flag, fw_quant_cap in fw_op_cap['quant'].items():
                    if quant_mode_flag not in op_cap['quant']:
                        op_cap['quant'][quant_mode_flag] = deepcopy(fw_quant_cap)
                    for item_name, item_options in op_user_cfg[key].items():
                        if item_options is not None and key in fw_quant_cap and item_name in fw_quant_cap[key]:
                            merged_options = []
                            for option in item_options:
                                if option in fw_quant_cap[key][item_name]:
                                    merged_options.append(option)
                                else:
                                    logger.warning("By default, {1}: {2} is not supported for {0} ".format(
                                                    key, item_name, option) + "in Intel Neural Compressor")
                                    logger.warning("Please visit the corresponding yaml file in " +
                                                   "neural_compressor/adaptor/ to enhance the default " +
                                                   "capability in Intel Neural Compressor")
                            if len(merged_options) == 0:
                                merged_options = fw_quant_cap[key][item_name]
                            op_cap['quant'][quant_mode_flag][key][item_name] = merged_options
        return op_cap

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
            
    def _parse_cap_helper_v3(self, cap):
        """

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
                        'fp32':{} # use []? (op_name, op_type): ('precision', ('fp32',)) or ('precision', ('fp32','fp32'))
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
            has_weight = False
            for op_cap in op_cap_lst:
                has_weight = 'weight' in op_cap
                attrs_lst = ['activation', 'weight'] if has_weight else ['activation']
                for att in attrs_lst:
                    # For quantization
                    if 'activation' in op_cap and 'quant_mode' in op_cap['activation']:
                        quant_mode = op_cap['activation']['quant_mode']
                        att_dtype = op_cap[att]['dtype'][0]
                        signed_flag, _data_type = extract_data_type(att_dtype)
                        for item_name, item_options in op_cap[att].items():
                            if item_name not in ['dtype', 'quant_mode']:
                                parsed_op_cap[quant_mode][att][_data_type][signed_flag][item_name] = item_options
                    # For precision
                    else:
                        att_dtype = op_cap[att]['dtype']
                        parsed_op_cap['precision'][att][att_dtype] = {'dtype': att_dtype}
            parsed_cap[op_name_type] = parsed_op_cap
        logger.info(f"Parsed cap ............")
        logger.info(parsed_cap)
        return parsed_cap
        
    def _parse_cap_helper(self, cap):
        """Parse the capability and convert it into internal structure.
        
        # for op support int8 dynamic ptq, int8 static ptq, fp32, and bf16
        ('op_name1','optype1'):[
            {
            'activation': {
                'dtype': ['int8'],
                'quant_mode': 'dynamic',
                'scheme': ['sym'],
                'granularity': ['per_channel', 'per_tensor'],
                'algorithm': ['minmax']},
            'weight': {
                'dtype': ['int8'],
                'scheme': ['sym'],
                'granularity': ['per_channel', 'per_tensor'],
                'algorithm': ['minmax']}
            },
            {
            'activation': {
                'dtype': ['int8'],
                'quant_mode': 'static',
                'scheme': ['sym'],
                'granularity': ['per_channel', 'per_tensor'],
                'algorithm': ['minmax']},
            'weight': {
                'dtype': ['int8'],
                'scheme': ['sym'],
                'granularity': ['per_channel', 'per_tensor'],
                'algorithm': ['minmax']}
            },
            {
            'activation': {
                'dtype': 'fp32'},
            'weight': {
                'dtype': 'fp32'},
            },
            {
            'activation': {
                'dtype': 'bf16'},
            'weight': {
                'dtype': 'bf16'},
            },
            ],
        """
        parsed_cap = OrderedDict()
        for op_name_type, op_cap_lst in cap.items():
            parsed_op_cap = {'precision': [], 'quant': OrderedDict()}
            for op_cap in op_cap_lst:
                if 'quant_mode' in op_cap['activation']:
                    quant_mode = op_cap['activation']['quant_mode']
                    quant_mode = quant_mode[0] if isinstance(quant_mode, list) else quant_mode
                    act_quant_flag =  op_cap['activation']['dtype'] != ['fp32']
                    # change the 'int8' to adapt the quantization with different numbers of bits in future
                    quant_mode_flag = (quant_mode, 'int8', act_quant_flag) 
                    parsed_op_cap['quant'][quant_mode_flag] = op_cap
                else:
                    if isinstance(op_cap['activation']['dtype'], list):
                        parsed_op_cap['precision'] += op_cap['activation']['dtype']
                    else:
                        parsed_op_cap['precision'].append(op_cap['activation']['dtype'])
            parsed_cap[op_name_type] = parsed_op_cap
            parsed_cap[op_name_type]['op_weight_flag'] = 'weight' in op_cap_lst[0]
        return parsed_cap
    
    def _create_tuning_space(self, capability, usr_cfg):
        """Create tuning space.
        
        step1. merge the capability with usr_cfg
        step2. create the tuning space
        :param capability:
        :param usr_cfg:
        :return:
        """
        #capability['op'] = self._parse_cap_helper(capability['op'])
        tmp_cap2 = self._parse_cap_helper_v3(deepcopy(capability['op']))
        capability['op'] = tmp_cap2
        if usr_cfg:
            logger.info(f"#############  Before merged with user cfg")
            logger.info(capability)
            self._merge_with_user_cfg(capability, usr_cfg['quantization'])
            logger.info(f"#############  After Merged with user cfg")
            logger.info(capability)
        #self._parse_capability(capability)

    def query_items_by_quant_mode(self, quant_mode):
        """Collect all op items that support the specific quantization/precision mode.
        
        Args:
            quant_mode (str): fp32/bf16/dynamic/static

        Returns:
            List: the list of op items
        """
        items_lst = []
        for _, op_item in self.op_items.items():
            for quant_item in op_item.options:
                if quant_mode == quant_item.name or quant_mode in quant_item.name:
                    if op_item not in items_lst:
                        items_lst.append(op_item)
                        break
        return items_lst
    
    def query_quant_mode_item(self, op_name_type, quant_mode):
        """Interface for query the quantization item.

        Args:
            op_name_type: (op_name, op_type)
            quant_mode: The quantization mode.

        Returns:
            Return queried quantization item.
        """
        op_item = self.op_items[op_name_type]
        for quant_item in op_item.options:
            if quant_mode == quant_item.name or quant_mode in quant_item.name:
                return quant_item

    def query_item_option(self, op_name_type, quant_mode, key, val):
        """Check if the option exit in the tuning item.

        Args:
            op_name_type: (op_name, op_type)
            quant_mode: The quantization mode.
            key: tuning item name.
            val: option of tuning item .

        Returns:
            bool: Return True if the option exit in the tuning item else False.
        """
        quant_mode_item = self.query_quant_mode_item(op_name_type, quant_mode)
        tuning_item = quant_mode_item.get_option_by_name(key)
        return tuning_item is not None and val in tuning_item.options
    
    def set_deafult_config(self, op_name_type, quant_mode):
        """Get the default tuning config.

        Args:
            op_name_type: (op_name, op_type)
            quant_mode: quantization mode.

        Returns:
            op_tuning_config: the default config according to the specified quantization mode.
        """
        from .tuning_structs import OpTuningConfig
        op_item = self.op_items[op_name_type]
        # set the first option as the default if the not support the required quant mode
        quant_mode_item = op_item.options[0]
        for quant_item in op_item.options:
            if quant_mode == quant_item.name or (isinstance(quant_mode, str) and quant_mode in quant_item.name):
                quant_mode_item = quant_item
                break
        # set the first option as the default for each tuning item
        config = {item.name: item.options[0] for item in quant_mode_item.options}
        op_tuning_config = OpTuningConfig(op_name_type[0], 
                                          op_name_type[1], 
                                          quant_mode, 
                                          self,
                                          config)
        return op_tuning_config

