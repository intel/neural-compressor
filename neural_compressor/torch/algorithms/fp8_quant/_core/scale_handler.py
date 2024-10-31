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

from .._quant_common.quant_config import ScaleFormat
from torch import Tensor, nn


def create_scale_tensor(orig_tensor, scale_format):
    if scale_format == ScaleFormat.CONST:
        if isinstance(orig_tensor, Tensor):
            return nn.Parameter(orig_tensor)
        elif isinstance(orig_tensor, list):
            return [nn.Parameter(x) for x in orig_tensor]
    elif scale_format == ScaleFormat.SCALAR:
        return scale_to_scalar(orig_tensor)
    else:
        raise ValueError("unexpected scale format value {}".format(scale_format))


# scalar scale is a performance optimization for LLM layers in small BS
def scale_to_scalar(scale):
    if isinstance(scale, Tensor):  # tensor case
        if scale.numel() == 1:
            return scale.item()
        else:
            raise Exception("scale as scalar isn't supported for scale tensors of dim > 0")
    elif isinstance(scale, float):  # already scalar case
        return scale
    else:
        raise Exception("unexpected scale instance type, expected Torch.tensor or float number")


def get_scale_dtype(scale):
    if isinstance(scale, Tensor):  # tensor case
        return scale.dtype
    elif isinstance(scale, float):  # already scalar case
        return type(scale).__name__
    else:
        raise Exception("unexpected scale instance type, expected Torch.tensor or float number")
