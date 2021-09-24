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
"""Configuration graph_optimization module."""

from typing import Any, Dict, List, Optional, Union

from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.json_serializer import JsonSerializer


class GraphOptimization(JsonSerializer):
    """Configuration Graph Optimization class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Graph Optimization class."""
        super().__init__()
        self.precisions: Optional[str] = None
        if data.get("precisions"):
            self.set_precisions(data.get("precisions"))  # type: ignore

        self.op_wise = None
        if isinstance(data.get("op_wise"), dict):
            self.op_wise = data.get("op_wise", {})

    def set_precisions(self, precisions: Union[str, List[str]]) -> None:
        """Update graph_optimization precisions in config."""
        if isinstance(precisions, str):
            self.precisions = precisions.replace(" ", "")
        elif isinstance(precisions, list):
            self.precisions = ",".join(precision.strip() for precision in precisions)
        else:
            raise ClientErrorException(
                "Precision should be comma separated string with "
                "precisions or list of string precisions",
            )
