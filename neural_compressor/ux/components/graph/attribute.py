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
"""Graph Node Attribute."""

from typing import Any

from neural_compressor.ux.utils.json_serializer import JsonSerializer


class Attribute(JsonSerializer):
    """Graph Node Attribute."""

    def __init__(self, name: str, attribute_type: str, value: Any):
        """Construct the object."""
        super().__init__()
        self.name = name
        self.attribute_type = attribute_type
        self.value = value
