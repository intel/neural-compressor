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
"""Configuration generator class."""
from typing import Any

from neural_compressor.ux.components.config_generator.config_generator import ConfigGenerator
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.utils.workload.evaluation import Accuracy, Evaluation, Metric
from neural_compressor.ux.utils.workload.graph_optimization import GraphOptimization


class GraphOptimizationConfigGenerator(ConfigGenerator):
    """Configuration generator class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize configuration generator."""
        super().__init__(*args, **kwargs)
        data = kwargs.get("data", {})
        self.optimization_precision: str = data["optimization_precision"]

    def generate(self) -> None:
        """Generate yaml config file."""
        config = Config()
        config.load(self.predefined_config_path)
        config.model = self.generate_model_config()
        config.evaluation = self.generate_evaluation_config()
        config.graph_optimization = self.generate_graph_optimization_config()
        config.dump(self.config_path)

    def generate_evaluation_config(self) -> Evaluation:
        """Generate evaluation configuration."""
        evaluation = Evaluation()
        evaluation.accuracy = Accuracy()

        if self.metric:
            evaluation.accuracy.metric = Metric(self.metric)

        evaluation.accuracy.dataloader = self.generate_dataloader_config(batch_size=1)
        return evaluation

    def generate_graph_optimization_config(self) -> GraphOptimization:
        """Generate graph optimization configuration."""
        graph_opt = GraphOptimization()
        graph_opt.precisions = self.optimization_precision
        return graph_opt
