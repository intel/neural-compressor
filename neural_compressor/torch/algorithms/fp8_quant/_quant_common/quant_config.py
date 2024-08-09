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

local_rank = int(os.getenv("LOCAL_RANK", "-1"))
world_size = int(os.getenv("WORLD_SIZE", "-1"))


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


class ScaleMethod(Enum):
    MAX = 1
    UNIT_SCALE = 2
    MAXABS_HW = 3
    MAXABS_POW2 = 4
    SMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2 = 5
    WEAKSMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2 = 6
    ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2 = 7
    ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2 = 8
    ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2 = 9
    ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2 = 10
    SMOOTHQUANT_OPT = 11
    MAXABS_HW_OPT_WEIGHT = 12
    MAXABS_POW2_OPT_WEIGHT = 13


def get_hqt_config(mod) -> Fp8cfg:
    return mod.__hqt_config__


def set_hqt_config(mod, config):
    mod.__hqt_config__ = config


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
                "types": ("torch.nn.Linear", "torch.nn.Conv2d", "BMM"),
            },  # types and names to be quantized. Allowlist by names is not yet implemented
            "mode": QuantMode.QUANTIZE,  # Quantize or Measure
            "scale_method": ScaleMethod.UNIT_SCALE,  # Method to quantize with
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
        }
        # assert measured_global_config['allowlist']['names'] == [''], "Allowlist names not yet implemented"

        # go over all user-defined keys from json, handle various cases
        for keys in custom_config:
            if keys == "mode":
                if custom_config[keys] == "NONE":
                    custom_config[keys] = QuantMode.NONE
                elif custom_config[keys] == "QUANTIZE":
                    custom_config[keys] = QuantMode.QUANTIZE
                elif custom_config[keys] == "MEASURE":
                    custom_config[keys] = QuantMode.MEASURE
                elif custom_config[keys] == "SHAPE":
                    custom_config[keys] = QuantMode.SHAPE
                else:
                    raise ValueError("invalid mode in custom config. Enter Quantize or Measure")

            if keys == "measure_exclude":
                if custom_config[keys] == "NONE":
                    custom_config[keys] = MeasureExclude.NONE
                elif custom_config[keys] == "OUTPUT":
                    custom_config[keys] = MeasureExclude.OUTPUT
                elif custom_config[keys] == "INPUT":
                    custom_config[keys] = MeasureExclude.INPUT
                elif custom_config[keys] == "ALL":
                    custom_config[keys] = MeasureExclude.ALL
                else:
                    raise ValueError("invalid measure exclude value in custom config. Enter OUTPUT or NONE")

            if keys == "fp8_config":
                if custom_config[keys].lower() == "e4m3":
                    custom_config[keys] = torch.float8_e4m3fn

                elif custom_config[keys].lower() == "e5m2":
                    custom_config[keys] = torch.float8_e5m2
                else:
                    raise ValueError("invalid fp8_config in custom config. Enter E4M3 or E5M2")

            if keys == "hp_dtype":
                if custom_config[keys].lower() == "bf16":
                    custom_config[keys] = torch.bfloat16
                elif custom_config[keys].lower() == "fp16":
                    custom_config[keys] = torch.float16
                elif custom_config[keys].lower() == "fp32":
                    custom_config[keys] = torch.float32
                else:
                    raise ValueError("invalid hp_dtype in custom config. Enter bf16, fp16 or fp32")

            if keys == "scale_method":
                if custom_config[keys].lower() == "unit_scale":
                    custom_config[keys] = ScaleMethod.UNIT_SCALE
                elif custom_config[keys].lower() == "max":
                    custom_config[keys] = ScaleMethod.MAX
                elif custom_config[keys].lower() == "maxabs_hw":
                    custom_config[keys] = ScaleMethod.MAXABS_HW
                elif custom_config[keys].lower() == "maxabs_pow2":
                    custom_config[keys] = ScaleMethod.MAXABS_POW2
                elif custom_config[keys].lower() == "maxabs_hw_opt_weight":
                    custom_config[keys] = ScaleMethod.MAXABS_HW_OPT_WEIGHT
                elif custom_config[keys].lower() == "maxabs_pow2_opt_weight":
                    custom_config[keys] = ScaleMethod.MAXABS_POW2_OPT_WEIGHT
                elif custom_config[keys].lower() == "smoothquant_weights_output_channel_maxabs_pow2":
                    custom_config[keys] = ScaleMethod.SMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2
                elif custom_config[keys].lower() == "weaksmoothquant_weights_output_channel_maxabs_pow2":
                    custom_config[keys] = ScaleMethod.WEAKSMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2
                elif custom_config[keys].lower() == "act_maxabs_hw_weights_pcs_maxabs_pow2":
                    custom_config[keys] = ScaleMethod.ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2
                elif custom_config[keys].lower() == "act_maxabs_hw_weights_pcs_opt_pow2":
                    custom_config[keys] = ScaleMethod.ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2
                elif custom_config[keys].lower() == "act_maxabs_pow2_weights_pcs_maxabs_pow2":
                    custom_config[keys] = ScaleMethod.ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2
                elif custom_config[keys].lower() == "act_maxabs_pow2_weights_pcs_opt_pow2":
                    custom_config[keys] = ScaleMethod.ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2
                elif custom_config[keys].lower() == "smoothquant_opt":
                    custom_config[keys] = ScaleMethod.SMOOTHQUANT_OPT
                else:
                    raise ValueError(
                        f'Invalid fp8_config in custom config ({custom_config[keys]}). should be in ["max", "unit_scale", "maxabs_hw", "maxabs_pow2", "maxabs_per_channel_pow2", "smoothquant_opt"]'
                    )

            if keys == "ignore_modules_wo_measures":
                custom_config[keys] = custom_config[keys].lower() == "true"

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

        base_name = measured_global_config["dump_stats_path"].split("/")[-1]
        folder_name = measured_global_config["dump_stats_path"][: -(len(base_name))]
        measured_global_config["dump_stats_base_path"] = folder_name
        os.makedirs(folder_name, exist_ok=True)
        worker_st = (
            ""
            if measured_global_config["local_rank"] is None
            else "_" + str(measured_global_config["local_rank"]) + "_" + str(measured_global_config["world_size"])
        )
        measured_global_config["shape_file"] = measured_global_config["dump_stats_path"] + "_hooks_shape" + worker_st
        measured_global_config["scale_file"] = (
            measured_global_config["dump_stats_path"]
            + "_hooks_"
            + measured_global_config["observer"]
            + "_"
            + measured_global_config["scale_method"].name
            + worker_st
        )
        if (measured_global_config["mode"] == QuantMode.MEASURE) or (
            measured_global_config["mode"] == QuantMode.QUANTIZE
        ):
            measured_global_config["measure_file"] = (
                measured_global_config["dump_stats_path"] + "_hooks_" + measured_global_config["observer"] + worker_st
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
        config_path = os.path.join(module_directory, "..", f"custom_config/{config_path}.json")
    try:
        logger.info("QUANT PACKAGE: Loading %s", config_path)
        with open(config_path) as config_json:
            config = json.load(config_json)
    except FileNotFoundError as e:
        raise Exception(f"Got exception: {e}. QUANT PACKAGE: Can't open {config_path}!")
    except JSONDecodeError as e:
        config_json.close()
        raise Exception(f"Got exception: {e}. QUANT PACKAGE: Can't load {config_path} json!")
    return config
