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
"""Save and load the quantized model."""


import json
import os

import torch

from neural_compressor.common.utils import load_config_mapping, save_config_mapping
from neural_compressor.torch.utils import QCONFIG_NAME, WEIGHT_NAME, logger


def save(model, example_inputs, output_dir="./saved_results"):
    """Save the quantized model and its configuration.

    Args:
        model (torch.nn.Module): The quantized model to be saved.
        example_inputs (torch.Tensor or tuple of torch.Tensor): Example inputs used for tracing the model.
        output_dir (str, optional): The directory where the saved results will be stored. Defaults to "./saved_results".
    """
    os.makedirs(output_dir, exist_ok=True)
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), QCONFIG_NAME)
    dynamic_shapes = model.dynamic_shapes
    quantized_ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    torch.export.save(quantized_ep, qmodel_file_path)
    for key, op_config in model.qconfig.items():
        model.qconfig[key] = op_config.to_dict()
    with open(qconfig_file_path, "w") as f:
        json.dump(model.qconfig, f, indent=4)

    logger.info("Save quantized model to {}.".format(qmodel_file_path))
    logger.info("Save configuration of quantized model to {}.".format(qconfig_file_path))


def load(output_dir="./saved_results"):
    """Load a quantized model from the specified output directory.

    Args:
        output_dir (str): The directory where the quantized model is saved. Defaults to "./saved_results".

    Returns:
        torch.nn.Module: The loaded quantized model.
    """
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    loaded_quantized_ep = torch.export.load(qmodel_file_path)
    return loaded_quantized_ep.module()
