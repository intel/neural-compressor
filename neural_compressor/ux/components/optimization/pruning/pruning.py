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
"""Pruning optimization class."""

import os
from typing import List

from neural_compressor.ux.components.config_generator.pruning_config_generator import (
    PruningConfigGenerator,
)
from neural_compressor.ux.components.optimization.optimization import Optimization
from neural_compressor.ux.components.optimization.pruning.optimize_model import optimize_model
from neural_compressor.ux.utils.exceptions import InternalException


class Pruning(Optimization):
    """Pruning optimization class."""

    def __init__(
        self,
        optimization_data: dict,
        project_data: dict,
        dataset_data: dict,
    ) -> None:
        """Initialize Pruning class."""
        super().__init__(optimization_data, project_data, dataset_data)
        self.joined_input_nodes: str = ""
        self.joined_output_nodes: str = ""
        if isinstance(self.input_nodes, list):
            self.joined_input_nodes = ",".join(self.input_nodes)
        if isinstance(self.output_nodes, list):
            self.joined_output_nodes = ",".join(self.output_nodes)

        if dataset_data["template_path"]:
            raise InternalException(
                "Custom datasets and metrics are not supported in Pruning.",
            )

    def execute(self) -> None:
        """Execute graph optimization."""
        if self.config_path is None:
            raise InternalException("Could not find path to config.")
        optimize_model(
            input_graph=self.input_graph,
            output_graph=self.output_graph,
            config=self.config_path,
            framework=self.framework,
        )

    @property
    def optimization_script(self) -> str:
        """Get optimization script path."""
        return os.path.join(
            os.path.dirname(__file__),
            "optimize_model.py",
        )

    @property
    def parameters(self) -> List[str]:
        """Get optimization parameters."""
        return [
            f"--config={self.config_path}",
            f"--input-graph={self.input_graph}",
            f"--output-graph={self.output_graph}",
            f"--framework={self.framework}",
        ]

    @property
    def configuration_data(self) -> dict:
        """Get configuration data for graph optimization config generator."""
        configuration_data: dict = super().configuration_data
        configuration_data.update(
            {
                "pruning_details": self.optimization.pruning_details,
            },
        )
        return configuration_data

    def generate_config(self) -> None:
        """Generate yaml config."""
        config_generator: PruningConfigGenerator = PruningConfigGenerator(
            workload_directory=self.workdir,
            configuration_path=self.config_path,
            data=self.configuration_data,
        )
        config_generator.generate()
