# -*- coding: utf-8 -*-
# Copyright (c) 2023 Intel Corporation
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
from neural_insights.components.graph.graph import Graph
from neural_insights.components.graph.reader.pytorch_reader import PyTorchReader
from neural_insights.components.model.model import Model
from neural_insights.utils.consts import Frameworks
from neural_insights.utils.utils import check_module, get_file_extension


class PyTorchModel(Model):
    """PyTorch Script Model class."""

    def get_model_graph(self) -> Graph:
        """Get model summary."""
        graph_reader = PyTorchReader(self)
        return graph_reader.read()

    def __init__(self, path: str) -> None:
        """Initialize object."""
        super().__init__(path)

    @staticmethod
    def get_framework_name() -> str:
        """Get the name of framework."""
        return Frameworks.PT.value

    @staticmethod
    def supports_path(path: str) -> bool:
        """Check if given path is of supported model."""
        return "pt" == get_file_extension(path)

    def guard_requirements_installed(self) -> None:
        """Ensure all requirements are installed."""
        check_module("torch")
