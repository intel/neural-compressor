#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
"""Tuning utility."""


from collections import OrderedDict

from deprecated import deprecated


@deprecated(version="2.0")
class OrderedDefaultDict(OrderedDict):
    """Ordered default dict."""

    def __missing__(self, key):
        """Initialize value for the missing key."""
        self[key] = value = OrderedDefaultDict()
        return value


@deprecated(version="2.0")
def extract_data_type(data_type: str) -> str:
    """Extract data type and signed from data type.

    Args:
        data_type: The original data type such as uint8, int8.

    Returns:
        (signed or unsigned, data type without signed)
    """
    return ("signed", data_type) if data_type[0] != "u" else ("unsigned", data_type[1:])


@deprecated(version="2.0")
def reverted_data_type(signed_flag: str, data_type: str) -> str:
    """Revert the data type."""
    return data_type if signed_flag == "signed" else "u" + data_type


@deprecated(version="2.0")
def get_adaptor_name(adaptor):
    """Get adaptor name.

    Args:
        adaptor: adaptor instance.
    """
    adaptor_name = type(adaptor).__name__.lower()
    adaptor_name_lst = ["onnx", "tensorflow", "pytorch"]
    for name in adaptor_name_lst:
        if adaptor_name.startswith(name):
            return name
    return ""
