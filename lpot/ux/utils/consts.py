# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constant values."""

from lpot.version import __version__ as lpot_version

github_info = {
    "user": "intel",
    "repository": "lpot",
    "tag": f"v{lpot_version}",
}


class Precisions:
    """Precisions enumeration."""

    INT8 = "int8"
    FP32 = "fp32"
    MIXED = "bf16,fp32"
