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
"""Data shape class."""
from typing import Any, Dict, List, Optional, Union

from neural_compressor.ux.utils.json_serializer import JsonSerializer


class Shape(JsonSerializer):
    """Data shape definition."""

    def __init__(self, shape: Optional[str] = "", trusted: bool = False) -> None:
        """Object construction."""
        super().__init__()

        self.shape = shape
        self.trusted = trusted

    def serialize(
        self,
        serialization_type: str = "default",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Serialize Shape class to dict."""
        result = {}
        for key, value in self.__dict__.items():
            if key in self._skip:
                continue
            result.update({key: value})
        return result
