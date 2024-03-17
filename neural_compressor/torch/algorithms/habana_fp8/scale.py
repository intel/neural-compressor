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

import habana_frameworks.torch.core as htcore
import torch

scale_method_mapping = {}


def scale_method_registry(name):
    def new_scale_method(scale_method_cls):
        global scale_method_mapping
        scale_method_mapping[name] = scale_method_cls
        return scale_method_cls

    return new_scale_method


@scale_method_registry("hw")
def hardware_scale_method(scale):
    scale_list = torch.tensor([16, 1, 1 / 16, 1 / 256])
    return torch.clip(
        2 ** (torch.ceil(torch.log2(scale) / 4) * 4),
        torch.tensor(scale_list[-1], dtype=scale.dtype, device=scale.device),
        torch.tensor(scale_list[0], dtype=scale.dtype, device=scale.device),
    )


@scale_method_registry("pow2")
def pow2_scale_method(scale):
    return 2 ** torch.ceil(torch.log2(scale))


@scale_method_registry("unit")
def unit_scale_method(scale):
    return torch.tensor(1.0)


@scale_method_registry("self")
def self_scale_method(scale):
    return scale


def map_gaudi_scale(scale, method):
    scale_method = scale_method_mapping[method]
    return scale_method(scale)
