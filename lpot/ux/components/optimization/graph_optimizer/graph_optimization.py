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
from typing import List

from lpot.ux.components.optimization.graph_optimizer.optimize_model import (
    optimize_graph,
    optimize_graph_config,
)
from lpot.ux.components.optimization.optimization import Optimization


class GraphOptimization(Optimization):
    """Optimization class."""

    def execute(self) -> None:
        """Execute graph optimization."""
        if self.tune:
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
                precisions=self.output_precision,
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
                    f"--input-nodes={self.input_nodes}",
                    f"--output-nodes={self.output_nodes}",
                    f"--precisions={self.output_precision}",
                ],
            )
        return parameters
