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
"""Optimization class."""

import os
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.hw_info import HWInfo
from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.utils.workload.workload import Workload


class Optimization(JsonSerializer):
    """Optimization class."""

    def __init__(
        self,
        workload: Workload,
        template_path: Optional[str] = None,
    ) -> None:
        """Initialize Optimization class."""
        self.input_graph: str = workload.model_path
        self.input_precision: str = workload.input_precision
        self.output_graph: str = workload.model_output_path
        self.output_precision: str = workload.output_precision
        self.framework: str = workload.framework

        self.input_nodes: Optional[str] = None
        self.output_nodes: Optional[str] = None
        self.set_boundary_nodes(workload)

        self.tune: bool = workload.tune
        self.config_path = workload.config_path

        self.instances: int = 1
        self.cores_per_instance: int = HWInfo().cores // self.instances

        if not os.path.exists(self.input_graph):
            raise ClientErrorException("Could not found model in specified path.")

        self._command: Optional[List[str]] = None

    def execute(self) -> None:
        """Execute optimization."""
        raise NotImplementedError

    @property
    def command(self) -> List[str]:
        """Get execution command."""
        if self._command is None:
            if isinstance(self.optimization_script, str) and isinstance(self.parameters, list):
                self._command = ["python", self.optimization_script] + self.parameters
            else:
                raise NotImplementedError
        return self._command

    @command.setter
    def command(self, value: List[str]) -> None:
        """Set optimization command."""
        if not isinstance(value, list):
            raise InternalException("Command should be a list.")
        self._command = value

    @property
    @abstractmethod
    def optimization_script(self) -> str:
        """Get optimization script path."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> List[str]:
        """Get optimization parameters."""
        raise NotImplementedError

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize Optimization class to dict."""
        result = {}
        for key, value in self.__dict__.items():
            variable_name = re.sub(r"^_", "", key)
            if variable_name == "command":
                result[variable_name] = " ".join(value)
            else:
                result[variable_name] = value
        return result

    def set_boundary_nodes(self, workload: Workload) -> None:
        """Set optimization input and output nodes according to workload."""
        if isinstance(workload.input_nodes, str):
            self.input_graph = workload.input_nodes
        if isinstance(workload.input_nodes, list):
            self.input_nodes = ",".join(workload.input_nodes)

        if isinstance(workload.output_nodes, str):
            self.output_nodes = workload.output_nodes
        if isinstance(workload.output_nodes, list):
            self.output_nodes = ",".join(workload.output_nodes)
