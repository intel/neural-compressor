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


# pylint:disable=import-error
import json
import os

import torch

try:
    import intel_extension_for_pytorch as ipex
except:
    assert False, "Please install IPEX for static quantization."

from neural_compressor.torch.utils import QCONFIG_NAME, WEIGHT_NAME, logger


def save(model, output_dir="./saved_results"):
    """Saves the quantized model to the output path.

    Args:
        model (RecursiveScriptModule): raw fp32 model or prepared model.
        output_dir (str, optional): output path to save the quantized model.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), QCONFIG_NAME)
    device = next(model.parameters(), None).device.type if next(model.parameters(), None) else "cpu"
    if device == "cpu":
        model.ori_save(qmodel_file_path)
        with open(qconfig_file_path, "w") as f:
            json.dump(model.tune_cfg, f, indent=4)
    else:  # pragma: no cover
        from neural_compressor.common.utils import save_config_mapping

        torch.jit.save(model, qmodel_file_path)
        save_config_mapping(model.qconfig, qconfig_file_path)

    logger.info("Save quantized model to {}.".format(qmodel_file_path))
    logger.info("Save configuration of quantized model to {}.".format(qconfig_file_path))


def load(output_dir="./saved_results"):
    """Loads the quantized model from the output path.

    Args:
        output_dir (str, optional): output path to load the quantized model.

    Returns:
        A quantized model.
    """
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    model = torch.jit.load(qmodel_file_path)
    model = torch.jit.freeze(model.eval())
    logger.info("Quantized model loading successful.")
    return model
