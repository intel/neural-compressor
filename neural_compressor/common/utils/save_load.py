#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""The module for save/load config."""

import json
import os


def save_config_mapping(config_mapping, qconfig_file_path):  # pragma: no cover
    """Save config mapping to json file.

    Args:
        config_mapping (dict): config mapping.
        qconfig_file_path (str): path to saved json file.
    """
    per_op_qconfig = {}
    for (op_name, op_type), op_config in config_mapping.items():
        value = {op_config.name: op_config.to_dict()}
        per_op_qconfig[str((op_name, op_type))] = value

    with open(qconfig_file_path, "w") as f:
        json.dump(per_op_qconfig, f, indent=4)


def load_config_mapping(qconfig_file_path, config_name_mapping):  # pragma: no cover
    """Reload config mapping from json file.

    Args:
        qconfig_file_path (str): path to saved json file.
        config_name_mapping (dict): map config name to config object.
                                    For example: ConfigRegistry.get_all_configs()["torch"]

    Returns:
        config_mapping (dict): config mapping.
    """
    config_mapping = {}
    with open(qconfig_file_path, "r") as f:
        per_op_qconfig = json.load(f)
    for key, value in per_op_qconfig.items():
        op_name, op_type = eval(key)
        # value here is a dict, so we convert it to an object with config_name_mapping,
        # which is defined in a specific framework.
        config_name = next(iter(value))
        config_obj = config_name_mapping[config_name]["cls"]()
        config_obj.from_dict(value[config_name])
        config_mapping[(op_name, op_type)] = config_obj
    return config_mapping
