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

import torch

from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator, INCAcceleratorType
from ..utils.logger import logger
from ..prepare_quant.prepare_model import get_world_size, get_local_rank
from .._core.scale_methods.scale_method_parser import parse_scale_method, validate_and_populate_scale_method, convert_scale_method_strings_to_enum
from .._core.scale_methods.scale_method_config import get_scale_method_from_config, check_scale_method_fields, ScaleMethodString, CfgStr, ScaleGranularity, ScaleValueType, ScaleRoundMethod


class QuantMode(Enum):
    NONE = 0
    QUANTIZE = 1
    MEASURE = 2
    SHAPE = 3
    LOAD = 4

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

class TrueFalse(Enum):
    TRUE = True
    FALSE = False


class ScaleFormat(Enum):
    CONST = 1  # scales is const and persistent tensor
    SCALAR = 2  # scales is non-const, non-persistent tensor with data ptr, used for low BS performance optimization


class DeviceForScalesType(Enum):
    GAUDI2 = INCAcceleratorType.GAUDI2
    GAUDI3 = INCAcceleratorType.GAUDI3


_config_to_enum = {
    "mode": QuantMode,
    "measure_exclude": MeasureExclude,
    "fp8_config": SupportedFp8,
    "hp_dtype": HpDtype,
    "scale_method": ScaleMethodString,
    "recalc_scales": TrueFalse,
    "ignore_modules_wo_measures": TrueFalse,
    "use_qdq": TrueFalse,
    "fake_quant": TrueFalse,
    "scale_format": ScaleFormat,
    "device_for_scales": DeviceForScalesType,
    "measure_on_hpu": TrueFalse,
    "dynamic_quantization": TrueFalse,
}


_configs_that_use_enum_value = [
    "fp8_config",
    "hp_dtype",
    "ignore_modules_wo_measures",
    "recalc_scales",
    "fake_quant",
    "use_qdq",
    "device_for_scales",
    "measure_on_hpu",
    "dynamic_quantization",
]

# TODO [SW-217813]: support dynamic quantization in all ops and remove
from neural_compressor.torch.algorithms.fp8_quant.model_configs import get_patched_module_table, ModuleInfo

def is_supported_dynamic_op(op_str):
    """
    Dynamically checks if the given op supports dynamic quantization
    by looking up its ModuleInfo and checking for a 'supports_dynamic_quantization' attribute.
    """
    patched_table = get_patched_module_table()
    info = patched_table.get(op_str)
    ret = getattr(info, "supports_dynamic_quantization", False) if info is not None else False
    logger.trace("Checking if %s is supported for dynamic quantization: %s", op_str, ret)
    return ret


def get_hqt_config(mod) -> Fp8cfg:
    return mod.__hqt_config__


def set_hqt_config(mod, config):
    mod.__hqt_config__ = config


def _get_enum_from_string(EnumClass, string, key):
    string = str(string)  # bool must be converted to string
    if not hasattr(EnumClass, string.upper()):
        raise ValueError(
            f"Invalid '{key}' value in custom config ('{string}'). Enter one of {[m.name for m in EnumClass]}"
        )
    return EnumClass[string.upper()]


def _validate_dump_path(dump_stats_path):
    dirname = os.path.dirname(dump_stats_path)
    basename = os.path.basename(dump_stats_path)
    if not os.access(dirname, os.W_OK):  # checks if the directory is not writable
        raise ValueError(f"Measurements dump directory '{dirname}' is non-writable")
    files_to_backup = [fname for fname in os.listdir(dirname) if fname.startswith(basename)]
    if files_to_backup:
        from datetime import datetime
        backup_dirname = f"backup_{basename}_{datetime.now().strftime('%d-%m_%H:%M:%S')}"
        try:
            os.mkdir(f"{dirname}/{backup_dirname}")
        except FileExistsError:
            pass
        for fname in files_to_backup:
            try:
                os.rename(f"{dirname}/{fname}", f"{dirname}/{backup_dirname}/{fname}")
            except (OSError, FileNotFoundError):
                pass


@dataclass
class Fp8cfg:
    cfg: Mapping[str, Any]

    def parse(custom_config: Mapping[str, str]) -> Fp8cfg:
        world_size = get_world_size()
        local_rank = get_local_rank()
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
            "fake_quant": False, # Fake or Real Quant, fake_quant only works for linear(PatchedLinear) and matmul(PatchedMatmul), usually used for training.
            "use_qdq": False, # QDQ or Real Quant, QDQ works for operators in helper_modules.py, usually used for inference.
            "scale_method": ScaleMethodString.MAXABS_HW,  # Method to quantize with
            "scale_params": {},  # scaling parameters that are different then the default ones
            "observer": "maxabs",  # Supported ['shape', 'maxabs', 'maxabs_per_channel', 'save']
            "mod_dict": {},
            "ignore_modules_wo_measures": False,  # Determines whether to fail quantization on modules without existing measures or not to quantize them
            "local_rank": local_rank if local_rank >= 0 else None,
            "global_rank": None,
            "world_size": world_size if world_size >= 0 else None,
            "seperate_measure_files": True,  # Determines whether to expect one or several measure files when using more than one gaudi
            "device_type": auto_detect_accelerator().get_inc_accelerator_type(),  # Determines device type: Gaudi2, Gaudi3...
            "device_for_scales": None,  # Overrides device type for scale: Gaudi2, Gaudi3... Enables using only G2 scales on G3
            "measure_exclude": MeasureExclude.OUTPUT,
            "recalc_scales": False,
            "scale_format": ScaleFormat.SCALAR,
            "measure_on_hpu": True,  # Determines whether to measure model on hpu device.
            "row_parallel_linear_allreduce_quantization" : False, # Turn on/off fp8 allreduce optimization detailed in SW-207602
            "dynamic_quantization" : False, # Turn on/off fp8 dynamic quantization
            "calibration_sample_interval" : 0 # number of samples to process before dumping measurements, 0 means no automatic dumping
        }
        # go over all user-defined keys from json, handle various cases
        for keys in custom_config:
            if keys in _config_to_enum.keys():
                if keys == "scale_method" and isinstance(custom_config[keys], dict):
                    # TODO: SW-230643 Add hash_id to file name in new scale method config
                    measured_global_config["recalc_scales"] = True
                    measured_global_config["scale_method"] = convert_scale_method_strings_to_enum(custom_config[keys])
                    continue
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
        measured_global_config["scale_method"] = parse_scale_method(measured_global_config["scale_method"])
        scale_method_config = measured_global_config["scale_method"]
        validate_and_populate_scale_method(scale_method_config)


        if auto_detect_accelerator().current_device_name() == "cpu":
            if not measured_global_config["use_qdq"]:
                raise ValueError("For FP8 quantization, only QDQ mode is supported on CPU device.")
            if measured_global_config["scale_format"] == ScaleFormat.CONST and \
                check_scale_method_fields(scale_method_config, granularity_weight=ScaleGranularity.PTS, reducer=any):
                    measured_global_config["scale_format"] = ScaleFormat.SCALAR
                    logger.warning(f"FP8 per-tensor quantization on CPU device requires 'scale_format = SCALAR'")

        # If seperate_measure_files is True (default value), then it is assumed that there are multiple distinct measure and scale files
        # and they are stored in / loaded from paths with the correct index as a suffix. Else, only one is searched for.
        measured_global_config["local_rank"] = (
            local_rank if local_rank >= 0 and custom_config.get("seperate_measure_files", True) else None
        )
        # set device_for_scales config for gaudi device only
        if measured_global_config["device_type"].value > INCAcceleratorType.GAUDI_MIN.value:
            logger.debug("setting device for scales config")
            Fp8cfg.set_gaudi_device_for_scales(custom_config, measured_global_config, scale_method_config)

        if measured_global_config["scale_format"] == ScaleFormat.SCALAR:
            if check_scale_method_fields(scale_method_config, granularity_weight=ScaleGranularity.PCS, reducer=any) or \
               check_scale_method_fields(scale_method_config, granularity_activation=ScaleGranularity.PCS, reducer=any):
                measured_global_config["scale_format"] = ScaleFormat.CONST
                logger.warning(f"Cannot use 'scale_format = SCALAR' when using PCQ (Per Channel Quantization), Reduced to 'CONST'.")
            if measured_global_config["fake_quant"]:
                measured_global_config["scale_format"] = ScaleFormat.CONST
                logger.warning(f"Cannot use 'scale_format = SCALAR' when using fake_quant. Reduced to 'CONST'.")
        quant_mode = measured_global_config["mode"]
        dynamic_quantization = measured_global_config["dynamic_quantization"]
        # TODO [SW-217814]: get dynamic methods in a better way, or support file handling in dynamic mode
        if dynamic_quantization:
            if auto_detect_accelerator().current_device_name() == "cpu":
                raise ValueError("Currently CPU device doesn't support dynamic quantization")
            logger.info(f"NOTE: Using dynamic scale method, only supported ops will be quantized.")
            if measured_global_config["scale_format"] == ScaleFormat.SCALAR:
                measured_global_config["scale_format"] = ScaleFormat.CONST
                logger.warning(f"Cannot use 'scale_format = SCALAR' when using dynamic quantization. Reduced to 'CONST'.")
            # TODO: "Linear only" in types still causes issues as llama7b quantizes also self_attn,
            # which should be blocked for some reason. We might then want to set measured_global_config["allowlist"]["types"] = supported_dynamic_ops
            # TODO [SW-222725]: support HW aligned rounding in dynamic quantization
            if check_scale_method_fields(scale_method_config, rounding_method_activation= ScaleRoundMethod.HW_ALIGNED, reducer=any) or \
                check_scale_method_fields(scale_method_config, scale_value_type_activation= ScaleValueType.FIXED_VALUE, scale_value_type_weight= ScaleValueType.FIXED_VALUE, reducer=any):
                raise ValueError(
                    f"Unsupported config: scale_method {scale_method_config} is not supported in dynamic quantization"
                )
            if quant_mode not in (QuantMode.QUANTIZE, QuantMode.LOAD):
                raise ValueError(f"Quantization mode is {quant_mode}, but dynamic quantization is only supported in QUANTIZE or LOAD mode")
            measured_global_config["use_stats_files"] = False
            #TODO [SW-224403]: enable dynamic quantization in row parallel allreduce
            if measured_global_config["row_parallel_linear_allreduce_quantization"]:
                raise ValueError(f"Dynamic quantization is not supported when using row_parallel_linear_allreduce_quantization")
        else:
            if check_scale_method_fields(scale_method_config, scale_method= ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW, reducer=any):
                raise ValueError(
                    f"Unsupported config: scale_method ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW is supported only in dynamic quantization"
                )

        if (dynamic_quantization or
            check_scale_method_fields(scale_method_config, scale_value_type_activation= ScaleValueType.FIXED_VALUE, scale_value_type_weight= ScaleValueType.FIXED_VALUE, reducer=all)) and \
            quant_mode in (QuantMode.QUANTIZE, QuantMode.LOAD):
                logger.debug(f"Quantization mode is QUANTIZE or LOAD and all scale_method are quant only so stats files won't be used")
                measured_global_config["use_stats_files"] = False
        else:
            measured_global_config["use_stats_files"] = True
            base_name = os.path.basename(measured_global_config["dump_stats_path"])
            folder_name = os.path.dirname(measured_global_config["dump_stats_path"])
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
                + get_scale_method_from_config(measured_global_config["scale_method"][CfgStr.DEFAULT]).name #TODO SW-230643 Add hash_id to file name in new scale method config
                + worker_st
            )
            if (quant_mode == QuantMode.MEASURE) or (
                quant_mode == QuantMode.QUANTIZE
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

        if measured_global_config["mode"] == QuantMode.MEASURE:
            _validate_dump_path(measured_global_config["dump_stats_path"])

        return Fp8cfg(cfg=measured_global_config)

    @staticmethod
    def set_gaudi_device_for_scales(custom_config, measured_global_config, scale_method):
        current_device_type = measured_global_config["device_type"]
        if current_device_type.value < INCAcceleratorType.GAUDI_MIN.value:
            raise ValueError("device for scales config is supported for only gaudi device line.")
        if custom_config.get("device_for_scales", None) is None:
            # Device for scales is the current device by default
            measured_global_config["device_for_scales"] = current_device_type

        elif measured_global_config["device_for_scales"] != measured_global_config["device_type"]:
            # Currently, only maxabs_hw is supported for a different device scales configuration
            if not check_scale_method_fields(scale_method_dict=scale_method, scale_method= ScaleMethodString.MAXABS_HW, reducer=all):
                raise ValueError(
                    f"Unsupported config: scale_method: {scale_method} for scale device overriding: {measured_global_config['device_for_scales']}"
                )
            if not (
                measured_global_config["device_for_scales"] == INCAcceleratorType.GAUDI2
                and measured_global_config["device_type"] == INCAcceleratorType.GAUDI3
            ):
                raise ValueError(f"Unsupported config: device_for_scales={measured_global_config['device_for_scales']} "
                                f"for device_type={measured_global_config['device_type']}")

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
