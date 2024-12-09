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

import copy
import json
import os
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Union

import torch

from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import Fp8cfg
from neural_compressor.torch.algorithms.fp8_quant.prepare_quant.prepare_model import finish_measurements


def save_calib_result(model):
    if hasattr(model, "__hqt_config__") and isinstance(model.__hqt_config__, Fp8cfg):
        # TODO SW-184714 modify hqt notation to inc notation once code is ported
        finish_measurements(model)
    else:
        raise NotImplementedError("Saving calibration results currently supported only in HPU.")


def update_mode(config_path, measure_step=False, quant_step=False):
    with open(config_path, "r") as file:
        config = json.load(file)

    if (measure_step and config.get("mode") == "MEASURE") or (quant_step and config.get("mode") == "QUANTIZE"):
        return config_path
    else:
        if measure_step:
            config["mode"] = "MEASURE"
        if quant_step:
            config["mode"] = "QUANTIZE"

        temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        temp_file_path = temp_file.name

        with open(temp_file_path, "w") as temp_file:
            json.dump(config, temp_file)

        return temp_file_path


def generate_model_info(model):
    mod_inst_info = namedtuple("ModInstInfo", ["name", "parent"])
    parent_child_mod_dict = {}

    def create_mod_info_recursion(parent):
        for name, mod in parent.named_children():
            parent_child_mod_dict[mod] = mod_inst_info(name=name, parent=parent)
            create_mod_info_recursion(mod)

    create_mod_info_recursion(model)
    return parent_child_mod_dict


def get_patched_mod_list():
    from ._core.common import mod_default_dict

    patched_mod_list = []
    for patched_mod in mod_default_dict.values():
        patched_mod_list.append(patched_mod.patched_module.__name__)
    return patched_mod_list


def restore_patched_module(patched_model):
    from neural_compressor.torch.algorithms.fp8_quant.utils import helper_mods

    patched_mod_list = get_patched_mod_list()

    parent_child_mod_dict = generate_model_info(patched_model)
    with torch.no_grad():
        for name, patched_mod in patched_model.named_modules():
            patched_mod_type_str = patched_mod.__class__.__name__
            if patched_mod_type_str in patched_mod_list:
                parent = parent_child_mod_dict[patched_mod].parent
                name = parent_child_mod_dict[patched_mod].name
                class_name_org = (
                    getattr(patched_mod, "class_name_org", None) or patched_mod.__class__.__name__.split("Patched")[-1]
                )
                patched_mod.__dict__.pop("forward", None)
                origin_mod = helper_mods[class_name_org](patched_mod)
                setattr(parent, name, origin_mod)


def with_patched_module(model):
    patched_mod_list = get_patched_mod_list()

    for name, mod in model.named_modules():
        mod_type = mod.__class__.__name__
        if mod_type in patched_mod_list:
            return True
    return False
