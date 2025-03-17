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
import torch

try:
    import intel_extension_for_pytorch as ipex
except:  # pragma: no cover
    assert False, "Please install IPEX for smooth quantization."

from neural_compressor.torch.algorithms.static_quant import load, save


def recover_model_from_json(model, json_file_path, example_inputs):  # pragma: no cover
    """Recover ipex model from JSON file.

    Args:
        model (object): fp32 model need to do quantization.
        json_file_path (json): configuration JSON file for ipex.
        example_inputs (tuple or torch.Tensor or dict): example inputs that will be passed to the ipex function.

    Returns:
        model (object): quantized model
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
