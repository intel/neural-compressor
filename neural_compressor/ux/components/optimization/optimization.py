# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
from abc import abstractmethod
from typing import List, Optional

from neural_compressor.ux.components.names_mapper.names_mapper import MappingDirection, NamesMapper
from neural_compressor.ux.utils.consts import WORKSPACE_LOCATION
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.hw_info import HWInfo
from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.utils.utils import get_file_extension, normalize_string


class Optimization(JsonSerializer):
    """Optimization class."""

    def __init__(
        self,
        optimization_data: dict,
        project_data: dict,
        dataset_data: dict,
    ) -> None:
        """Initialize Optimization class."""
        super().__init__()
        self._skip.extend(["optimization", "dataloader"])
        names_mapper = NamesMapper(MappingDirection.ToCore)
        self.id: int = optimization_data["id"]
        self.project_id: int = project_data["id"]
        self.project_name: str = project_data["name"]

        self.optimization: OptimizationInterface = OptimizationInterface(optimization_data)

        self.dataloader: DataloaderInterface = DataloaderInterface(dataset_data)

        self.tune: bool = False

        self.model_id = project_data["input_model"]["id"]
        self.model_name = project_data["input_model"]["name"]
        self.input_graph: str = project_data["input_model"]["path"]
        self.model_domain: str = project_data["input_model"]["domain"]["name"]
        self.model_domain_flavour: str = project_data["input_model"]["domain_flavour"]["name"]
        self.input_nodes: Optional[List[str]] = project_data["input_model"]["input_nodes"]
        self.output_nodes: Optional[List[str]] = project_data["input_model"]["output_nodes"]
        self.input_precision: str = project_data["input_model"]["precision"]["name"]
        normalized_model_name = self.output_model_dir
        self.output_graph: str = os.path.join(normalized_model_name, self.output_model_name)
        self.output_precision = self.optimization.output_precision
        self.framework: str = names_mapper.map_name(
            parameter_type="framework",
            value=project_data["input_model"]["framework"]["name"],
        )

        self.config_path: Optional[str] = os.path.join(
            self.workdir,
            self.config_filename,
        )

        self.instances: int = 1
        self.cores_per_instance: int = HWInfo().cores // self.instances

        if not os.path.exists(self.input_graph):
            raise ClientErrorException("Could not found model in specified path.")

        self._command: Optional[List[str]] = None

    def execute(self) -> None:
        """Execute optimization."""
        raise NotImplementedError

    @property
    def output_model_name(self) -> str:
        """Get output model name."""
        output_name = normalize_string(self.optimization.name)
        if os.path.isfile(self.input_graph):
            output_name += "." + get_file_extension(self.input_graph)

        return output_name

    @property
    def output_model_dir(self) -> str:
        """Get path to output model directory."""
        normalized_project_name = normalize_string(self.project_name)
        normalized_optimization_name = normalize_string(self.optimization.name)
        return os.path.join(
            WORKSPACE_LOCATION,
            f"{normalized_project_name}_{self.project_id}",
            "optimizations",
            f"{normalized_optimization_name}_{self.optimization.id}",
        )

    @property
    def workdir(self) -> str:
        """Get path to optimization workdir."""
        normalized_project_name = normalize_string(self.project_name)
        normalized_optimization_name = normalize_string(self.optimization.name)
        return os.path.join(
            WORKSPACE_LOCATION,
            f"{normalized_project_name}_{self.project_id}",
            "optimizations",
            f"{normalized_optimization_name}_{self.optimization.id}",
        )

    @property
    def config_filename(self) -> str:
        """Get filename for configuration."""
        return "config.yaml"

    @property
    def configuration_data(self) -> dict:
        """Get configuration data for config generator."""
        configuration_data = {
            "framework": self.framework,
            "model": {
                "name": self.model_name,
                "domain": self.model_domain,
                "domain_flavour": self.model_domain_flavour,
                "input_graph": self.input_graph,
                "input_nodes": self.input_nodes,
                "output_nodes": self.output_nodes,
            },
            "batch_size": self.optimization.batch_size,
            "dataloader": {
                "dataset": {self.dataloader.dataset_type: self.dataloader.dataset_params},
                "transforms": self.dataloader.transforms,
                "filter": self.dataloader.filter,
                "metric": self.dataloader.metric,
            },
            "dataset_type": self.dataloader.dataset_type,
        }
        return configuration_data

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

    @abstractmethod
    def generate_config(self) -> None:
        """Generate yaml config."""
        raise NotImplementedError


class OptimizationInterface:
    """Interface for optimization."""

    id: int
    name: str
    type: str
    output_precision: str
    batch_size: int
    sampling_size: int

    def __init__(self, optimization_data: dict):
        """Initialize optimization interface with data."""
        names_mapper = NamesMapper(MappingDirection.ToCore)
        self.id = optimization_data["id"]
        self.name = optimization_data["name"]
        self.type = optimization_data["optimization_type"]["name"]
        self.batch_size = optimization_data["batch_size"]
        self.output_precision = names_mapper.map_name(
            parameter_type="precision",
            value=optimization_data["precision"]["name"],
        )


class DataloaderInterface:
    """Interface for dataloader."""

    dataset_type: str
    dataset_params: dict
    transforms: dict
    filter: dict
    metric: dict

    def __init__(self, dataloader_data: dict):
        """Initialize optimization interface with data."""
        self.dataset_type = dataloader_data["dataset_type"]
        self.dataset_params = dataloader_data["parameters"]
        self.transforms = dataloader_data["transforms"]
        self.filter = dataloader_data["filter"]
        self.metric = dataloader_data["metric"]
