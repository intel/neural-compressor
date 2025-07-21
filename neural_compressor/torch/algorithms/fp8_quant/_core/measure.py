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

import numpy as np
import torch

from abc import abstractmethod

from .._quant_common.quant_config import MeasureExclude, QuantMode, get_hqt_config, set_hqt_config
from .save_measure import gmod_list
from .scale_methods.scale_method_config import ScaleMethodString
from ..utils.logger import logger
from .common import load_file, save_file, ShapeList
from .patching_common import generate_model_info, mod_default_dict, mod_types, parent_child_mod_dict
from ..model_configs import ModuleExtraConfig
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator
from neural_compressor.torch.algorithms.fp8_quant.model_configs import (
    OBSERVER_TYPES,
    OBSERVER_PARAMS,
    IMOD_DICT,
)
from neural_compressor.torch.algorithms.fp8_quant._core.common import dequant_original_fp8_weight_if_needed


cur_accelerator = auto_detect_accelerator()


def patch_module_measure(mod, mconfig, mod_dict):
    """Replaces the module with patched module according to mconfig.

    Args:
        mod (nn.module): The module that will be replaced with patched module that measures the inputs.
        mconfig (e.g. MaxAbsObserver/MaxAbsPerChannelObserver): The observer object that will measure the parameters.
        mod_dict (dict): dictionary from module name to its patched module.

    Returns:
        nn.module: The new module after patching.
    """
    parent = parent_child_mod_dict[mod].parent
    name = parent_child_mod_dict[mod].name
    parent = parent_child_mod_dict[mod].parent
    patched_mod = mod_dict[mod.__class__.__name__].patched_module(mod, parent, mconfig, name)
    setattr(parent, name, patched_mod)
    return patched_mod


def init_measure_object(mod, name, observer_class, mod_type, skip_measure_output, d_shape=None, params=None):
    input_observer = [
        observer_class(
            "input%d" % (i),
            mod,
            None if d_shape is None else d_shape.inputs[i],
            params=None if params is None else params.inputs[i],
        )
        for i in range(mod_type.num_inputs)
    ]
    if (not mod_type.required_output) and skip_measure_output:
        # excluding output measurements
        output_observer = None
    else:
        output_observer = [
            observer_class(
                "output%d" % (i),
                mod,
                None if d_shape is None else d_shape.outputs,
                params=None if params is None else params.outputs,
            )
            for i in range(mod_type.num_outputs)
        ]
    params_observer = {
        param_name: observer_class(
            param_name,
            mod,
            None if d_shape is None else d_shape.params[param_name],
            params=None if params is None else params.params[param_name],
        )
        for param_name in mod_type.param_names
    }
    measure_object = ModuleExtraConfig(input_observer, output_observer, params_observer, None, params)
    return measure_object


def prepare_model(model, mod_list=None):
    """Defines the observer class and modules for measurement as preparation.

    Args:
        model (nn.module): The model that will be measured.
        mod_list (list, optional): The specific submodules that will be measured in the model. Defaults to None.
    """
    config = get_hqt_config(model).cfg
    observer_class = OBSERVER_TYPES[config["observer"]]
    if (config.get("shape_file", None) is not None) and (observer_class != OBSERVER_TYPES["shape"]):
        shapes_fname = config["shape_file"] + ".json"
        d_shapes = load_file(shapes_fname, ShapeList, False)
    else:
        d_shapes = None
    # TODO SW-220992: Add dependency check for observer, just like maxabs_per_channel is dependent on shape
    if not d_shapes and observer_class == OBSERVER_TYPES["maxabs_per_channel"]:
        raise RuntimeError("Required shape files are missing from measurement directory")
    gmod_list.extend(mod_list)
    generate_model_info(model)
    register_patched_measure_modules(model, mod_list, observer_class, d_shapes)

def setup_calibration_counter(model, config):
    # used for automatically dumping measurements
    calibration_sample_interval = int(config["calibration_sample_interval"])
    if calibration_sample_interval > 0:
        from .save_measure import add_calibration_samples_counter
        add_calibration_samples_counter(model, calibration_sample_interval)

def register_patched_measure_modules(model, mod_list, observer_class, d_shapes=None):
    """Replace the submodules of the model that appear in mod_list with a patched submodule that uses the given observer_class
    so the submodule will perform measurement on inputs/outputs in forward stage.
    Weights measurement is done during model preparation as they are static.

    Args:
        model (nn.module): The model that will be measured.
        mod_list (list): The specific submodules that will be measured in the model.
        observer_class (e.g. MaxAbsObserver/MaxAbsPerChannelObserver): The observer type that will measure the weights.
        d_shapes (dict, optional): Defaults to None.
    """
    top_level_config = get_hqt_config(model)
    config = top_level_config.cfg
    setup_calibration_counter(model, config)
    skip_outputs_measurements = config["measure_exclude"] & (MeasureExclude.OUTPUT | MeasureExclude.ALL)
    patched_types = set()
    non_patched_types = set()
    patched_modules = []
    with torch.no_grad():
        for name, mod in model.named_modules():
            if (name in mod_list) or (mod_list is None):
                IMOD_DICT[mod] = name
                mod_type_str = mod.__class__.__name__
                mod_type = config["mod_dict"][mod_type_str]
                params = (
                    OBSERVER_PARAMS[config["observer"]][mod_type]
                    if (config["observer"] in OBSERVER_PARAMS) and (mod_type in OBSERVER_PARAMS[config["observer"]])
                    else None
                )
                patched_types.add(type(mod))

                set_hqt_config(mod, top_level_config)  # set config in the module, as it consumed by the patched module
                if mod_type == "dynamic_moe" and hasattr(mod, "num_experts"):
                    # override default number of outputs for dynamic moe
                    mod_types[mod_type].num_outputs = mod.num_experts+1
                    logger.warning(f"Dynamic moe num_outputs set to {mod.num_experts+1}")
                mod_extra_config = (
                    init_measure_object(
                        mod,
                        name,
                        observer_class,
                        mod_types[mod_type],
                        skip_outputs_measurements,
                        (d_shapes[name] if ((d_shapes is not None) and (name in d_shapes)) else None),
                        params,
                    )
                    if mod_default_dict[mod_type_str].should_measure_and_quant
                    else None
                )
                pmod = patch_module_measure(mod, mod_extra_config, mod_default_dict)
                if pmod._mod_extra_config:
                    for param_name in pmod._mod_extra_config.params:
                        param = getattr(pmod, param_name)
                        param = dequant_original_fp8_weight_if_needed(pmod.orig_mod, param)
                        if config["measure_on_hpu"]:
                            param = param.to(cur_accelerator.name())
                        pmod._mod_extra_config.params[param_name].measure(param)
                        cur_accelerator.synchronize()
                if observer_class == OBSERVER_TYPES["save"]:
                    save_module(pmod)
                patched_modules.append(name)
            else:
                non_patched_types.add(type(mod))
    logger.debug("Patched module types: %s", patched_types)
    logger.debug("None-patched module types: %s", non_patched_types)
    logger.debug("Patched modules: %s", patched_modules)
    logger.debug("Total patched modules: %d", len(patched_modules))
    if config["measure_on_hpu"]:
        model = model.to(cur_accelerator.name())
    cur_accelerator.synchronize()


def load_measurements(model, fname):
    config = get_hqt_config(model).cfg
    source_fname = fname if fname is not None else config["measure_file"]
    fname_np = source_fname + ".npz"
    d = load_file(
        fname_np,
        np.ndarray,
        fail_on_file_not_exist=(config["scale_method"] != ScaleMethodString.UNIT_SCALE),
    )
    from collections import defaultdict

    d = defaultdict(lambda: None, d)

    return d


def save_json(d, fname):
    with open(fname, "w") as f:
        json.dump(d, f, indent=4)


def load_json(fname):
    with open(fname, "r") as f:
        d = json.load(f)
    return d


def save_module(mod):
    folder_name = os.path.join(mod.config["dump_stats_base_path"], "tensors")
    os.makedirs(folder_name, exist_ok=True)
    file_base_name = os.path.join(folder_name, IMOD_DICT[mod] + "_module.pt")
    torch.save(mod.state_dict(), file_base_name)
