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

import json
import os

import habana_frameworks.torch.core as htcore
import numpy as np
import torch

from abc import abstractmethod

from .._quant_common.quant_config import MeasureExclude, QuantMode, ScaleMethod, get_hqt_config, set_hqt_config
from ..utils.logger import logger
from .common import *
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator
from neural_compressor.torch.algorithms.fp8_quant.model_configs import (
    OBSERVER_TYPES,
    OBSERVER_PARAMS,
    IMOD_DICT,
)
cur_accelerator = auto_detect_accelerator()


gmod_list = []


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
    patched_mod = mod_dict[mod.__class__.__name__].patched_module(mod, mconfig, name)
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
    gmod_list.extend(mod_list)
    generate_model_info(model)
    register_patched_measure_modules(model, mod_list, observer_class, d_shapes)


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


def is_measure_done(mod_extra_config):
    # check if measurements were collected by observer
    for obs in ([] if mod_extra_config.inputs is None else mod_extra_config.inputs) + (
        [] if mod_extra_config.outputs is None else mod_extra_config.outputs
    ):
        if obs.is_used():
            return True
    return False


def get_mod_extra_config_dict(model):
    mcd = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "_mod_extra_config") and mod._mod_extra_config:
            if is_measure_done(mod._mod_extra_config):
                name = name.replace("_orig_mod.", "")  # remove _orig_mod part added by dynamo mechanism
                mcd[name] = mod._mod_extra_config
            else:
                logger.debug(
                    "Layer '%s' has no measurements therefore it can't be quantized during quantization.",
                    name,
                )
    return mcd


def measure_control_to_state_dict(mcd):
    sd = {}
    sdl = {}
    for mname in mcd:
        sd[mname] = dict()
        sdl[mname] = dict()
        sd[mname]["inputs"] = [
            mcd[mname].inputs[i].state.detach().cpu().float().numpy()
            for i in range(len(mcd[mname].inputs))
            if mcd[mname].inputs[i].state is not None
        ]
        sdl[mname]["inputs"] = [
            mcd[mname].inputs[i].state.detach().cpu().float().numpy().tolist()
            for i in range(len(mcd[mname].inputs))
            if mcd[mname].inputs[i].state is not None
        ]
        if mcd[mname].outputs:
            sd[mname]["outputs"] = [
                mcd[mname].outputs[i].state.detach().cpu().float().numpy()
                for i in range(len(mcd[mname].outputs))
                if mcd[mname].outputs[i].state is not None
            ]
            sdl[mname]["outputs"] = [
                mcd[mname].outputs[i].state.detach().cpu().float().numpy().tolist()
                for i in range(len(mcd[mname].outputs))
                if mcd[mname].outputs[i].state is not None
            ]
        if len(mcd[mname].params) > 0:
            sd[mname]["params"] = dict()
            sdl[mname]["params"] = dict()
            for param_name in mcd[mname].params:
                if mcd[mname].params[param_name].state is not None:
                    sd[mname]["params"][param_name] = mcd[mname].params[param_name].state.detach().cpu().float().numpy()
                    sdl[mname]["params"][param_name] = (
                        mcd[mname].params[param_name].state.detach().cpu().float().numpy().tolist()
                    )
    return sd, sdl


def save_measurements(model, fname=None):
    config = get_hqt_config(model).cfg
    if config["mode"] in [QuantMode.MEASURE, QuantMode.SHAPE]:
        if fname is None:
            if ("measure_file" in config) and (config["measure_file"] is not None):
                fname_base = config["measure_file"]
                measure_type = "DynamicRange"
            elif ("shape_file" in config) and (config["shape_file"] is not None) and (config["observer"] == "shape"):
                fname_base = config["shape_file"]
                measure_type = "Shape"
            fname_np = fname_base + ".npz"
            fname_list = fname_base + ".json"
        else:
            logger.warning("'fname' is not None - Measurements/Shapes will not be saved")
            return
        mcd = get_mod_extra_config_dict(model)
        sd, sdl = measure_control_to_state_dict(mcd)

        logger.info("Dumping measurements")
        save_file(model, sd, np.ndarray, fname_np, measure_type)
        save_file(model, sdl, list, fname_list, measure_type)
        save_json(gmod_list, fname_base + "_mod_list.json")


def load_measurements(model, fname):
    config = get_hqt_config(model).cfg
    source_fname = fname if fname is not None else config["measure_file"]
    fname_np = source_fname + ".npz"
    d = load_file(
        fname_np,
        np.ndarray,
        fail_on_file_not_exist=(config["scale_method"] != ScaleMethod.UNIT_SCALE),
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
