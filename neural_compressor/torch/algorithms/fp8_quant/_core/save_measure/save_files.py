# Copyright (c) 2025 Intel Corporation
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

from neural_compressor.torch.algorithms.fp8_quant.utils.logger import logger
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import get_hqt_config, QuantMode
from neural_compressor.torch.algorithms.fp8_quant._core.common import save_file, save_json


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

def create_files_names(config, fname = None):
    if fname is None:
        if ("measure_file" in config) and (config["measure_file"] is not None):
            fname_base = config["measure_file"]
            measure_type = "DynamicRange"
        elif ("shape_file" in config) and (config["shape_file"] is not None) and (config["observer"] == "shape"):
            fname_base = config["shape_file"]
            measure_type = "Shape"
        fname_np = fname_base + ".npz"
        fname_list = fname_base + ".json"
        return fname_base, fname_np, fname_list, measure_type
    else:
        logger.warning("'fname' is not None - Measurements/Shapes will not be saved")
        return

def save_measurements_files(model, state_dict, state_list, gmod_list, fname_np, fname_list, fname_base, measure_type,
                            num_samples=0):
    import numpy as np
    logger.info("Dumping measurements")
    save_file(model, state_dict, np.ndarray, fname_np, measure_type, num_samples)
    save_file(model, state_list, list, fname_list, measure_type, num_samples)
    save_json(gmod_list, fname_base + "_mod_list.json")
    return


gmod_list = [] # global list extened with patched modules in measure.prepare_model


def save_measurements(model, fname=None):
    config = get_hqt_config(model).cfg
    if config["mode"] in [QuantMode.MEASURE, QuantMode.SHAPE]:
        fname_base, fname_np, fname_list, measure_type = create_files_names(config, fname)
        mcd = get_mod_extra_config_dict(model)
        sd, sdl = measure_control_to_state_dict(mcd)
        num_samples = model.calibration_samples_counter if hasattr(model, "calibration_samples_counter") else 0
        save_measurements_files(model, sd, sdl, gmod_list, fname_np, fname_list, fname_base, measure_type, num_samples)
