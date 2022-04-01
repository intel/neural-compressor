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
"""Profiling class."""

import os
from typing import List, Optional

from neural_compressor.ux.components.config_generator.profiling_config_generator import (
    ProfilingConfigGenerator,
)
from neural_compressor.ux.components.names_mapper.names_mapper import MappingDirection, NamesMapper
from neural_compressor.ux.components.profiling.factory import ProfilerFactory
from neural_compressor.ux.components.profiling.profiler import Profiler
from neural_compressor.ux.utils.consts import WORKSPACE_LOCATION
from neural_compressor.ux.utils.utils import normalize_string


class Profiling:
    """Profiling class."""

    def __init__(
        self,
        project_data: dict,
        profiling_data: dict,
    ) -> None:
        """Initialize Profiling class."""
        names_mapper = NamesMapper(MappingDirection.ToCore)
        self.project_id: int = project_data["id"]
        self.project_name: str = project_data["name"]

        self.profiling_id: int = profiling_data["id"]
        self.profiling_name: str = profiling_data["name"]

        self.batch_size: int = 1

        self.model_id: int = profiling_data["model"]["id"]
        self.model_name: str = profiling_data["model"]["name"]
        self.model_path: str = profiling_data["model"]["path"]
        self.model_domain: str = profiling_data["model"]["domain"]["name"]
        self.model_domain_flavour: str = profiling_data["model"]["domain_flavour"]["name"]
        self.input_nodes: List[str] = profiling_data["model"]["input_nodes"]
        self.output_nodes: List[str] = profiling_data["model"]["output_nodes"]

        self.dataloader: DataloaderInterface = DataloaderInterface(profiling_data["dataset"])

        self.framework: str = names_mapper.map_name(
            parameter_type="framework",
            value=profiling_data["model"]["framework"]["name"],
        )

        self.num_threads: int = profiling_data["num_threads"]

        self.profiling_script = os.path.join(
            os.path.dirname(__file__),
            "profile_model.py",
        )

        self.command = [
            "python",
            self.profiling_script,
            "--profiling-config",
            self.config_path,
        ]

    def execute(self) -> None:
        """Execute profiling and collect results."""
        profiler: Optional[Profiler] = ProfilerFactory.get_profiler(
            profiling_data=self.profiling_data,
        )
        if profiler is None:
            raise Exception("Could not find profiler for workload.")
        profiler.profile_model()

    @property
    def workdir(self) -> str:
        """Get path to optimization workdir."""
        normalized_project_name = normalize_string(self.project_name)
        normalized_profiling_name = normalize_string(self.profiling_name)
        normalized_model_name = normalize_string(self.model_name)
        return os.path.join(
            WORKSPACE_LOCATION,
            f"{normalized_project_name}_{self.project_id}",
            "models",
            f"{normalized_model_name}_{self.model_id}",
            "profilings",
            f"{normalized_profiling_name}_{self.profiling_id}",
        )

    @property
    def config_filename(self) -> str:
        """Get profiling config filename."""
        return "config.json"

    @property
    def config_path(self) -> str:
        """Get profiling config path."""
        return os.path.join(
            self.workdir,
            self.config_filename,
        )

    def generate_config(self) -> None:
        """Generate json config for profiling."""
        config_generator = ProfilingConfigGenerator(
            workload_directory=self.workdir,
            configuration_path=self.config_path,
            data=self.profiling_data,
        )
        config_generator.generate()

    @property
    def profiling_data(self) -> dict:
        """Collect profiling data for profiling config generator."""
        profiling_data = {
            "framework": self.framework,
            "model": {
                "name": self.model_name,
                "domain": self.model_domain,
                "domain_flavour": self.model_domain_flavour,
                "input_graph": self.model_path,
                "input_nodes": self.input_nodes,
                "output_nodes": self.output_nodes,
            },
            "batch_size": self.batch_size,
            "dataloader": {
                "dataset": {self.dataloader.dataset_type: self.dataloader.dataset_params},
                "transforms": self.dataloader.transforms,
                "filter": self.dataloader.filter,
                "metric": self.dataloader.metric,
            },
            "num_threads": self.num_threads,
            "dataset_type": self.dataloader.dataset_type,
        }
        return profiling_data


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
