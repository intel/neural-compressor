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
                op = f"('{op_name}', '{v}')"
            else:
                value[k] = v
        per_op_qconfig[op] = value

    with open(qconfig_file_path, "w") as f:
        json.dump(per_op_qconfig, f, indent=4)


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
