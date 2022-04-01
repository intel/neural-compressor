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
"""Optimization class."""

import os
from typing import List

from neural_compressor.ux.components.config_generator.graph_optimization_config_generator import (
    GraphOptimizationConfigGenerator,
)
from neural_compressor.ux.components.optimization.graph_optimizer.optimize_model import (
    optimize_graph,
    optimize_graph_config,
)
from neural_compressor.ux.components.optimization.optimization import Optimization
from neural_compressor.ux.utils.exceptions import InternalException


class GraphOptimization(Optimization):
    """Optimization class."""

    def __init__(
        self,
        optimization_data: dict,
        project_data: dict,
        dataset_data: dict,
    ) -> None:
        """Initialize Optimization class."""
        super().__init__(optimization_data, project_data, dataset_data)
        self.joined_input_nodes: str = ""
        self.joined_output_nodes: str = ""
        if isinstance(self.input_nodes, list):
            self.joined_input_nodes = ",".join(self.input_nodes)
        if isinstance(self.output_nodes, list):
            self.joined_output_nodes = ",".join(self.output_nodes)

        if not self.tune:
            self.config_path = None

        if dataset_data["template_path"]:
            raise InternalException(
                "Custom datasets and metrics are not supported in Graph Optimization.",
            )

    def execute(self) -> None:
        """Execute graph optimization."""
        if self.tune:
            if self.config_path is None:
                raise InternalException("Could not find path to config.")
            optimize_graph_config(
                input_graph=self.input_graph,
                output_graph=self.output_graph,
                config=self.config_path,
                framework=self.framework,
            )
        else:
            optimize_graph(
                input_graph=self.input_graph,
                output_graph=self.output_graph,
                input=self.input_nodes,
                output=self.output_nodes,
                framework=self.framework,
                precisions=self.optimization.output_precision,
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
        parameters = [
            f"--input-graph={self.input_graph}",
            f"--output-graph={self.output_graph}",
            f"--framework={self.framework}",
        ]

        if self.tune:
            parameters.extend(
                [
                    f"--config={self.config_path}",
                ],
            )
        else:
            parameters.extend(
                [
                    f"--input-nodes={self.joined_input_nodes}",
                    f"--output-nodes={self.joined_output_nodes}",
                    f"--precisions={self.optimization.output_precision}",
                ],
            )
        return parameters

    @property
    def configuration_data(self) -> dict:
        """Get configuration data for graph optimization config generator."""
        configuration_data: dict = super().configuration_data
        configuration_data.update(
            {
                "optimization_precision": self.optimization.output_precision,
            },
        )
        return configuration_data

    def generate_config(self) -> None:
        """Generate yaml config."""
        if self.tune:
            config_generator: GraphOptimizationConfigGenerator = GraphOptimizationConfigGenerator(
                workload_directory=self.workdir,
                configuration_path=self.config_path,
                data=self.configuration_data,
            )
            config_generator.generate()
