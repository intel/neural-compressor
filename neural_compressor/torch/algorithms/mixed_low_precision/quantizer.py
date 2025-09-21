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

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.algorithms.mixed_low_precision.modules import HPUMixedPrecisionLinear
from neural_compressor.torch.algorithms.weight_only.modules import HPUWeightOnlyLinear


class HybridGPTQQuantizer(Quantizer):
    def __init__(self, quant_config):
        super().__init__(quant_config)
        if isinstance(quant_config, dict):
            json_file = [cfg.json_file for cfg in quant_config.values()]
            assert len(json_file) > 0, "Cannot get json file from config."
            self.quant_config = json_file[0]

    def prepare(self, model):
        _prepare(model)
        return model

    def convert(self, model):
        _convert(model)
        return model


def set_module(model, op_name, new_module):
    """Set module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.
        new_module (object): the input model.

    Returns:
        module (object).
    """
    module = model
    name_list = op_name.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    setattr(module, name_list[-1], new_module)


def _convert(model):
    for name, module in model.named_modules():
        # replace `HPUWeightOnlyLinear`s forward func
        if isinstance(module, HPUWeightOnlyLinear):
            module = HPUMixedPrecisionLinear.convert_from_weight_only(module)
            set_module(model, name, module)

    return model


def _prepare(model):
    for name, module in model.named_modules():
        # replace `HPUWeightOnlyLinear`s forward func
        if isinstance(module, HPUWeightOnlyLinear):
            module = HPUMixedPrecisionLinear.prepare_from_weight_only(module)
            set_module(model, name, module)

    return model
