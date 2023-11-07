# Copyright (c) 2023 Intel Corporation
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

from neural_compressor.common.config import BaseConfig
from neural_compressor.common.utility import print_nested_dict
from neural_compressor.torch.quantization.config import parse_config_from_dict


def quantize(model, quant_config):
    if isinstance(quant_config, dict):
        qconfig = parse_config_from_dict(quant_config)
        print("parsed qconfig from dict: ")
        print_nested_dict(qconfig.to_dict())
        qconfig.to_json_file("parsed_from_dict.json")
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), "Please pass a dict or config instance as the quantization configuration."
    return model
