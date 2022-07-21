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
"""Benchmark class."""

import os
from typing import List

from neural_compressor.ux.components.benchmark.benchmark_model import benchmark_model
from neural_compressor.ux.components.config_generator.benchmark_config_generator import (
    BenchmarkConfigGenerator,
)
from neural_compressor.ux.components.names_mapper.names_mapper import MappingDirection, NamesMapper
from neural_compressor.ux.utils.consts import WORKSPACE_LOCATION
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.utils import normalize_string


class Benchmark:
    """Benchmark class."""

    def __init__(
        self,
        project_data: dict,
        benchmark_data: dict,
    ) -> None:
        """Initialize Benchmark class."""
        names_mapper = NamesMapper(MappingDirection.ToCore)
        self.project_id: str = project_data["id"]
        self.project_name: str = project_data["name"]
        self.original_model_path: str = project_data["input_model"]["path"]

        self.benchmark_id: str = benchmark_data["id"]
        self.benchmark_name: str = benchmark_data["name"]

        self.dataloader: DataloaderInterface = DataloaderInterface(benchmark_data["dataset"])

        self.instances: int = benchmark_data["number_of_instance"]
        self.cores_per_instance = benchmark_data["cores_per_instance"]

        self.model_id = benchmark_data["model"]["id"]
        self.model_name = benchmark_data["model"]["name"]
        self.model_path = benchmark_data["model"]["path"]
        self.model_domain: str = benchmark_data["model"]["domain"]["name"]
        self.model_domain_flavour: str = benchmark_data["model"]["domain_flavour"]["name"]
        self.input_nodes: List[str] = benchmark_data["model"]["input_nodes"]
        self.output_nodes: List[str] = benchmark_data["model"]["output_nodes"]

        self.precision = benchmark_data["model"]["precision"]["name"]
        self.warmup_iterations = benchmark_data["warmup_iterations"]
        self.iterations = benchmark_data["iterations"]
        self.num_of_instance = benchmark_data["number_of_instance"]
        self.cores_per_instance = benchmark_data["cores_per_instance"]
        self.batch_size = benchmark_data["batch_size"]
        self.mode = benchmark_data["mode"]
        self.command = benchmark_data["execution_command"]

        self.framework: str = names_mapper.map_name(
            parameter_type="framework",
            value=benchmark_data["model"]["framework"]["name"],
        )
        self.config_path: str = os.path.join(
            self.workdir,
            self.config_filename,
        )

        if not os.path.exists(self.model_path):
            raise ClientErrorException(
                f"Could not find model at specified path: {self.model_path}.",
            )

        self.benchmark_script = os.path.join(
            os.path.dirname(__file__),
            "benchmark_model.py",
        )

        if self.framework != "pytorch":
            self.command = [
                "python",
                self.benchmark_script,
                "--config",
                self.config_path,
                "--input-graph",
                self.model_path,
                "--mode",
                self.mode,
                "--framework",
                self.framework,
            ]

    def execute(self) -> None:
        """Execute benchmark and collect results."""
        benchmark_model(
            input_graph=self.model_path,
            config=self.config_path,
            benchmark_mode=self.mode,
            framework=self.framework,
        )

    @property
    def workdir(self) -> str:
        """Get path to optimization workdir."""
        normalized_project_name = normalize_string(self.project_name)
        normalized_benchmark_name = normalize_string(self.benchmark_name)
        normalized_model_name = normalize_string(self.model_name)
        return os.path.join(
            WORKSPACE_LOCATION,
            f"{normalized_project_name}_{self.project_id}",
            "models",
            f"{normalized_model_name}_{self.model_id}",
            "benchmarks",
            f"{normalized_benchmark_name}_{self.benchmark_id}",
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
            "mode": self.mode,
            "warmup_iterations": self.warmup_iterations,
            "iterations": self.iterations,
            "number_of_instance": self.num_of_instance,
            "cores_per_instance": self.cores_per_instance,
            "dataset_type": self.dataloader.dataset_type,
        }
        return configuration_data

    def generate_config(self) -> None:
        """Generate yaml config for benchmark."""
        config_generator: BenchmarkConfigGenerator = BenchmarkConfigGenerator(
            workload_directory=self.workdir,
            configuration_path=self.config_path,
            data=self.configuration_data,
        )
        config_generator.generate()


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
