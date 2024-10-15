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

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum, Flag, auto
from json.decoder import JSONDecodeError
from typing import Any, Mapping

import habana_frameworks.torch.utils.experimental as htexp
import torch

from ..utils.logger import logger

try:
    world_size = torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()
except:
    world_size = -1
    local_rank = -1


class QuantMode(Enum):
    NONE = 0
    QUANTIZE = 1
    MEASURE = 2
    SHAPE = 3


class MeasureExclude(Flag):
    NONE = auto()
    INPUT = auto()
    OUTPUT = auto()
    PARAMS = auto()
    ALL = auto()


class SupportedFp8(Enum):
    E4M3 = torch.float8_e4m3fn
    E5M2 = torch.float8_e5m2


class HpDtype(Enum):
    BF16 = torch.bfloat16
    FP16 = torch.float16
    FP32 = torch.float32


class ScaleMethod(Enum):
    MAX = 1
    UNIT_SCALE = 2
    HW_ALIGNED_SINGLE_SCALE = 3
    MAXABS_HW = 4
    MAXABS_POW2 = 5
    SMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2 = 6
    WEAKSMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2 = 7
    ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2 = 8
    ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2 = 9
    ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2 = 10
    ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2 = 11
    SMOOTHQUANT_OPT = 12
    MAXABS_HW_OPT_WEIGHT = 13
    MAXABS_POW2_OPT_WEIGHT = 14


class TrueFalse(Enum):
    TRUE = True
    FALSE = False


class ScaleFormat(Enum):
    CONST = 1  # scales is const and persistent tensor
    SCALAR = 2  # scales is non-const, non-persistent tensor with data ptr, used for low BS performance optimization


_config_to_enum = {
    "mode": QuantMode,
    "measure_exclude": MeasureExclude,
    "fp8_config": SupportedFp8,
    "hp_dtype": HpDtype,
    "scale_method": ScaleMethod,
    "recalc_scales": TrueFalse,
    "ignore_modules_wo_measures": TrueFalse,
    "fake_quant": TrueFalse,
    "scale_format": ScaleFormat,
}


_configs_that_use_enum_value = ["fp8_config", "hp_dtype", "ignore_modules_wo_measures", "recalc_scales", "fake_quant"]
_scale_methods_quant_only = [ScaleMethod.UNIT_SCALE, ScaleMethod.HW_ALIGNED_SINGLE_SCALE]


def get_hqt_config(mod) -> Fp8cfg:
    return mod.__hqt_config__


def set_hqt_config(mod, config):
    mod.__hqt_config__ = config


def _get_enum_from_string(EnumClass, str, key):
    if not hasattr(EnumClass, str.upper()):
        raise ValueError(
            f"Invalid '{key}' value in custom config ('{str}'). Enter one of {[m.name for m in EnumClass]}"
        )
    return EnumClass[str.upper()]


@dataclass
class Fp8cfg:
    cfg: Mapping[str, Any]

    def parse(custom_config: Mapping[str, str]) -> Fp8cfg:
        measured_global_config = {
            "dump_stats_path": "stats",
            "fp8_config": torch.float8_e4m3fn,  # The parameters of the chosen Quantization methed
            "hp_dtype": torch.bfloat16,  # The parameters of the chosen Quantization methed
            "blocklist": {
                "names": [],
                "types": (),
            },  # types and names to not be quantized
            "allowlist": {
                "names": [],
                "types": (),
            },  # types and names to be quantized. Allowlist by names is not yet implemented
            "mode": QuantMode.QUANTIZE,  # Quantize or Measure
            "fake_quant": False,  # Fake or Real Quant
            "scale_method": ScaleMethod.MAXABS_HW,  # Method to quantize with
            "scale_params": {},  # scaling parameters that are different then the default ones
            "observer": "maxabs",  # Supported ['shape', 'maxabs', 'maxabs_per_channel', 'save']
            "mod_dict": {},
            "ignore_modules_wo_measures": False,  # Determines whether to fail quantization on modules without existing measures or not to quantize them
            "local_rank": local_rank if local_rank >= 0 else None,
            "global_rank": None,
            "world_size": world_size if world_size >= 0 else None,
            "seperate_measure_files": True,  # Determines whether to expect one or several measure files when using more than one gaudi
            "device_type": htexp._get_device_type(),  # Determines device type: Gaudi2, Gaudi3...
            "measure_exclude": MeasureExclude.OUTPUT,
            "recalc_scales": False,
            "scale_format": ScaleFormat.CONST,
        }
        # assert measured_global_config['allowlist']['names'] == [''], "Allowlist names not yet implemented"

        # go over all user-defined keys from json, handle various cases
        for keys in custom_config:
            if keys in _config_to_enum.keys():
                custom_config[keys] = _get_enum_from_string(_config_to_enum[keys], custom_config[keys], keys)
                if keys in _configs_that_use_enum_value:
                    custom_config[keys] = custom_config[keys].value

            # TODO [SW-175936] - remove checking for old key names whitelist and blacklist.
            if isinstance(custom_config[keys], dict):
                for keys_2 in custom_config[keys]:
                    if keys == "whitelist":
                        measured_global_config["allowlist"][keys_2] = custom_config[keys][keys_2]
                    elif keys == "blacklist":
                        measured_global_config["blocklist"][keys_2] = custom_config[keys][keys_2]
                    else:
                        measured_global_config[keys][keys_2] = custom_config[keys][keys_2]
            else:
                if keys == "whitelist":
                    measured_global_config["allowlist"] = custom_config[keys]
                elif keys == "blacklist":
                    measured_global_config["blocklist"] = custom_config[keys]
                else:
                    measured_global_config[keys] = custom_config[keys]

        # If seperate_measure_files is True (default value), then it is assumed that there are multiple distinct measure and scale files
        # and they are stored in / loaded from paths with the correct index as a suffix. Else, only one is searched for.
        measured_global_config["local_rank"] = (
            local_rank if local_rank >= 0 and custom_config.get("seperate_measure_files", True) else None
        )

        scale_method = measured_global_config["scale_method"]
        quant_mode = measured_global_config["mode"]
        if scale_method in _scale_methods_quant_only:
            if quant_mode == QuantMode.QUANTIZE:
                logger.debug(
                    f"Quantization mode is quant, scale_method is {scale_method}, so stats files won't be used"
                )
                measured_global_config["use_stats_files"] = False
            else:
                raise ValueError(
                    f"Quantization mode is {quant_mode}, scale_method is {scale_method} (quant only). Unexpected behavior. "
                    "This scale method doesn't require measurements."
                )
        else:
            measured_global_config["use_stats_files"] = True
            base_name = measured_global_config["dump_stats_path"].split("/")[-1]
            folder_name = measured_global_config["dump_stats_path"][: -(len(base_name))]
            measured_global_config["dump_stats_base_path"] = folder_name
            os.makedirs(folder_name, exist_ok=True)
            worker_st = (
                ""
                if measured_global_config["local_rank"] is None
                else "_" + str(measured_global_config["local_rank"]) + "_" + str(measured_global_config["world_size"])
            )
            measured_global_config["shape_file"] = (
                measured_global_config["dump_stats_path"] + "_hooks_shape" + worker_st
            )
            measured_global_config["scale_file"] = (
                measured_global_config["dump_stats_path"]
                + "_hooks_"
                + measured_global_config["observer"]
                + "_"
                + scale_method.name
                + worker_st
            )
            if (quant_mode == QuantMode.MEASURE) or (quant_mode == QuantMode.QUANTIZE):
                measured_global_config["measure_file"] = (
                    measured_global_config["dump_stats_path"]
                    + "_hooks_"
                    + measured_global_config["observer"]
                    + worker_st
                )
            # measured_global_config['dump_stats_path'] += '_hooks_.json'

            logger.debug("HQT Paths:")
            logger.debug("base_name='%s'", base_name)
            logger.debug("folder_name='%s'", folder_name)
            logger.debug(
                "measured_global_config['shape_file']='%s'",
                measured_global_config["shape_file"],
            )
            logger.debug(
                "measured_global_config['scale_file']='%s'",
                measured_global_config["scale_file"],
            )
            if "measure_file" in measured_global_config.keys():
                logger.debug(
                    "measured_global_config['measure_file']='%s'",
                    measured_global_config["measure_file"],
                )
            logger.debug(
                "measured_global_config['dump_stats_path']='%s'",
                measured_global_config["dump_stats_path"],
            )

        return Fp8cfg(cfg=measured_global_config)


def _read_config_from_file(config_path: str) -> Mapping[str, str]:
    logger.debug("QUANT PACKAGE: using %s config", config_path)

    module_directory = os.path.dirname(os.path.abspath(__file__))

    # if file in absolute path doesn't exist, try looking in cfg directory
    if not os.path.isfile(config_path):
        config_path = os.path.join(module_directory, "..", f"custom_config/{config_path}")
    try:
        logger.info("QUANT PACKAGE: Loading %s", config_path)
        with open(config_path) as config_json:
            config = json.load(config_json)
    except FileNotFoundError as e:
        raise Exception(f"Got exception: {e}. QUANT PACKAGE: Can't open {config_path}!")
    except JSONDecodeError as e:
        config_json.close()
        raise Exception(f"Got exception: {e}. QUANT PACKAGE: Can't load {config_path}!")
    return config
