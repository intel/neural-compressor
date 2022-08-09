# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022 Intel Corporation
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
"""PyTorch model class."""
import re
from typing import List

from neural_compressor.ux.components.model.model import Model
from neural_compressor.ux.utils.consts import Frameworks
from neural_compressor.ux.utils.utils import check_module, get_file_extension


class PyTorchScriptModel(Model):
    """PyTorch Script Model class."""

    def __init__(self, path: str) -> None:
        """Initialize object."""
        super().__init__(path)

    @staticmethod
    def _has_any_name_parts(nodes: set, name_parts: List[str]) -> bool:
        """Check if there is at least one node for name_parts."""
        matching_names = []
        for node in nodes:
            for partial_name in name_parts:
                search = re.match(partial_name, node)
                if search:
                    matching_names.append(node)
        return bool(matching_names)

    @staticmethod
    def get_framework_name() -> str:
        """Get the name of framework."""
        return Frameworks.PT.value

    @staticmethod
    def supports_path(path: str) -> bool:
        """Check if given path is of supported model."""
        return "py" == get_file_extension(path)

    def guard_requirements_installed(self) -> None:
        """Ensure all requirements are installed."""
        check_module("torch")
