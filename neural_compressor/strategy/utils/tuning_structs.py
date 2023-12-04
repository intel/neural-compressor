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

from typing import Dict

from .constant import PRECISION_LIST, TUNING_ITEMS_LST, WEIGHT_ONLY_TUNING_ITEMS_LST


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
        self.op_quant_mode = op_quant_mode  # static/dynamic/fp32/bf16/fp16
        self.kwargs = kwargs
        self.act_dtype = None
        self.weight_dtype = None
        self.has_weight = self.op_name_type in tuning_space.ops_attr["weight"]
        self._set_dtype()
        self.tune_list = WEIGHT_ONLY_TUNING_ITEMS_LST if self.op_quant_mode == "weight_only" else TUNING_ITEMS_LST

    def _set_dtype(self):
        """Set the date type."""
        if self.op_quant_mode in PRECISION_LIST:
            self.act_dtype, self.weight_dtype = self.op_quant_mode, self.op_quant_mode
        else:
            self.act_dtype = self.kwargs.get("activation_dtype", None)
            if ("weight", "dtype") in self.kwargs:
                self.weight_dtype = self.kwargs[("weight", "dtype")]
            else:
                self.weight_dtype = self.kwargs.get("weight_dtype", None)

        assert self.act_dtype and isinstance(self.act_dtype, str), (
            f"Didn't assign the activation data type for {self.op_name, self.op_type}",
            f"with quant_mode {self.op_quant_mode}",
        )
        # if self.has_weight:
        #     assert self.weight_dtype, \
        #         (f"Didn't assign the weight data type for {self.op_name, self.op_type}", \
        #             f"with quant_mode {self.op_quant_mode}")

    def __repr__(self) -> str:
        """Display the tuning config as string.

        Returns:
            msg: the tuning config as string.
        """
        msg = f"op name: {self.op_name}, op type : {self.op_type} \n"
        msg += f"\t activation dtype: {self.act_dtype} \n"
        if self.op_quant_mode != "weight_only":
            # weight_dtype is contained in self.tune_list
            msg += f"\t weight dtype: {self.weight_dtype} \n" if self.has_weight else ""
        for key, val in self.kwargs.items():
            if key in self.tune_list:
                msg += f"\t {key[0]} {key[1]}: {val}\n"
        return msg

    def get_state(self):
        """Return the op tuning configuration.

        Returns:
            Dict: The op tuning state.
        """
        result = {}
        if self.has_weight:
            result["weight"] = {
                "dtype": self.weight_dtype,
            }
        result["activation"] = {
            "dtype": self.act_dtype,
            "quant_mode": self.op_quant_mode,
        }
        for key, val in self.kwargs.items():
            if key in self.tune_list:
                result[key[0]][key[1]] = val
        return result

    @classmethod
    def from_state(cls, config: Dict):
        """Create the tuning config from dict.

        Args:
            config: A dict includes the tuning config.
        """
        cls(**config)
