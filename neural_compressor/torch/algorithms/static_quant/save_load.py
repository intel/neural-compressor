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

# pylint:disable=import-error
import json
import os

import torch

try:
    import intel_extension_for_pytorch as ipex
except:
    assert False, "Please install IPEX for static quantization."

from neural_compressor.torch.utils import QCONFIG_NAME, WEIGHT_NAME, logger


def save_config_mapping(config_mapping, qconfig_file_path):
    """Save config mapping to json file.

    Args:
        config_mapping (dict): config mapping.
        qconfig_file_path (str): path to saved json file.
    """
    per_op_qconfig = {}
    for op_name, q_op_infos in config_mapping.items():
        value = {}
        for k, v in q_op_infos.items():
            if k == "op_type":
                op = (op_name, str(v))
            else:
                value[k] = v
        per_op_qconfig[str(op)] = value

    with open(qconfig_file_path, "w") as f:
        json.dump(per_op_qconfig, f, indent=4)


def recover_model_from_json(model, json_file_path, example_inputs):
    """Recover ipex model from JSON file.

    Args:
        model (object): fp32 model need to do quantization.
        json_file_path (json): configuration JSON file for ipex.
        example_inputs (tuple or torch.Tensor or dict): example inputs that will be passed to the ipex function.

    Returns:
        (object): quantized model
    """
    from torch.ao.quantization.observer import MinMaxObserver

    if ipex.__version__ >= "2.1.100":
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5, act_observer=MinMaxObserver)
    else:
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5, act_observer=MinMaxObserver())
    if isinstance(example_inputs, dict):
        model = ipex.quantization.prepare(model, qconfig, example_kwarg_inputs=example_inputs, inplace=True)
    else:
        model = ipex.quantization.prepare(model, qconfig, example_inputs=example_inputs, inplace=True)
    model.load_qconf_summary(qconf_summary=json_file_path)
    model = ipex.quantization.convert(model, inplace=True)
    with torch.no_grad():
        try:
            if isinstance(example_inputs, dict):
                # pylint: disable=E1120,E1123
                model = torch.jit.trace(model, example_kwarg_inputs=example_inputs)
            else:
                model = torch.jit.trace(model, example_inputs)
            model = torch.jit.freeze(model.eval())
        except:
            if isinstance(example_inputs, dict):
                # pylint: disable=E1120,E1123
                model = torch.jit.trace(model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False)
            else:
                model = torch.jit.trace(model, example_inputs, strict=False)
            model = torch.jit.freeze(model.eval())
        if isinstance(example_inputs, dict):
            model(**example_inputs)
            model(**example_inputs)
        elif isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
            model(*example_inputs)
            model(*example_inputs)
        else:
            model(example_inputs)
            model(example_inputs)
    return model


def save(model, output_dir="./saved_results"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), QCONFIG_NAME)
    save_config_mapping(model.tune_cfg[" "]["q_op_infos"], qconfig_file_path)
    model.save(qmodel_file_path)

    logger.info("Save quantized model to {}.".format(qmodel_file_path))
    logger.info("Save configuration of quantized model to {}.".format(qconfig_file_path))


def load(output_dir="./saved_results"):
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    model = torch.jit.load(qmodel_file_path)
    logger.info("Quantized model loading successful.")
    return model
