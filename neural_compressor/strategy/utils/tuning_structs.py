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

"""Tuning structure."""

import os
from typing import Dict, Any
from .constant import QUANT_MODE_SET
from ...utils import logger

class OpTuningConfig:
    """Op tuning config."""
    
    def __init__(self, op_name, op_type, op_quant_mode, tuning_space, kwargs={}):
        """Create the tuning config.

        Args:
            op_name: op name.
            op_type: op type.
            op_quant_mode: quantization mode.
            tuning_space: tuning space.
            kwargs: other parameters. Defaults to {}.
        """
        self.op_name = op_name
        self.op_type = op_type
        self.op_name_type = (self.op_name, self.op_type)
        self.op_quant_mode = op_quant_mode  # [static, dynamic]
        self.kwargs = kwargs
        self.has_weight = self.op_name_type in tuning_space.ops_attr['weight']
        self._get_dtype(tuning_space, op_quant_mode)
        
    def _get_dtype(self, tuning_space, path):
        full_path = tuning_space._get_op_default_path(self.op_name_type, path)
        self.act_dtype = tuning_space.ops_data_type[self.op_name_type][full_path['activation']]
        if self.has_weight:
            self.weight_dtype = tuning_space.ops_data_type[self.op_name_type][full_path['weight']]
        
    
    def _set_dtype(self, tuning_space, quant_bit='int8', act_quant_flag=None):
        #TODO remove it
        quant_mode = self.op_quant_mode
        op_name_type = (self.op_name, self.op_type)
        if quant_mode in QUANT_MODE_SET:
            for act_quant_flag in [True, False]:
                new_quant_mode = (quant_mode, quant_bit, act_quant_flag)
                if new_quant_mode in tuning_space.ops_dtype[op_name_type]:
                    quant_mode = new_quant_mode
                    break
        if quant_mode in tuning_space.ops_dtype[op_name_type]:
            self.act_dtype = tuning_space.ops_dtype[op_name_type][quant_mode]['act_dtype']
            q_dtype = tuning_space.ops_dtype[op_name_type][quant_mode]
            self.weight_dtype = q_dtype['weight_dtype'] if 'weight_dtype' in q_dtype else None
        else:
            first_quant_mode = list(tuning_space.ops_dtype[op_name_type].keys())[0]
            logger.debug(f"The op {op_name_type} does not support {quant_mode}, \
                         set as {first_quant_mode} instead.")
            quant_mode = first_quant_mode
            self.act_dtype = tuning_space.ops_dtype[op_name_type][quant_mode]['act_dtype']
            q_dtype = tuning_space.ops_dtype[op_name_type][quant_mode]
            self.weight_dtype = q_dtype['weight_dtype'] if 'weight_dtype' in q_dtype else None


    def __str__(self) -> str:
        """Display the tuning config as string.

        Returns:
            msg: the tuning config as string.
        """
        msg =  f"op name: {self.op_name}, op type : {self.op_type} \n"
        msg += f"\t activation dtype: {self.act_dtype} \n"
        msg += f"\t weight dtype: {self.weight_dtype} \n"  if self.has_weight else ""
        for key, val in self.kwargs.items():
            msg += f"\t {key[0]} {key[1]}: {val}\n"
        return msg

    def get_state(self):
        """Return the op tuning configuration.
        
        Returns:
            Dict: The op tuning state.
        """
        result = {}
        if self.has_weight:
            result['weight'] = {
                'dtype': self.weight_dtype,
            }
        result['activation'] = {
                'dtype': self.act_dtype,
                'quant_mode': self.op_quant_mode if isinstance(self.op_quant_mode, str) else self.op_quant_mode[0],
            }
        for key, val in self.kwargs.items():
            result[key[0]][key[1]] = val
        return result

    @classmethod
    def from_state(cls, config: Dict):
        """Create the tuning config from dict.

        Args:
            config: A dict includes the tuning config.
        """
        cls(**config)
