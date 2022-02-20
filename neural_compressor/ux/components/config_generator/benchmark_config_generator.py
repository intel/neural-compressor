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

from neural_compressor.ux.components.benchmark import Benchmarks
from neural_compressor.ux.components.config_generator.config_generator import ConfigGenerator
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.utils.workload.evaluation import (
    Accuracy,
    Evaluation,
    Metric,
    Performance,
)


class BenchmarkConfigGenerator(ConfigGenerator):
    """Configuration generator class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize configuration generator."""
        super().__init__(*args, **kwargs)
        data = kwargs.get("data", {})
        self.mode: str = data["mode"]
        self.warmup_iterations: int = data["warmup_iterations"]
        self.iterations: int = data["iterations"]
        self.num_of_instance: int = data["number_of_instance"]
        self.cores_per_instance: int = data["cores_per_instance"]

    def generate(self) -> None:
        """Generate yaml config file."""
        config = Config()
        config.load(self.predefined_config_path)
        config.quantization = None
        config.model = self.generate_model_config()
        config.evaluation = self.generate_evaluation_config()
        config.dump(self.config_path)

    def generate_evaluation_config(self) -> Evaluation:
        """Generate evaluation configuration."""
        evaluation = Evaluation()
        if self.mode == Benchmarks.ACC:
            evaluation.accuracy = self.generate_accuracy_config()
            evaluation.set_accuracy_postprocess_transforms(self.transforms)

        if self.mode == Benchmarks.PERF:
            evaluation.performance = self.generate_performance_config()

        return evaluation

    def generate_accuracy_config(self) -> Accuracy:
        """Generate accuracy evaluation configuration."""
        accuracy = Accuracy()
        if self.metric:
            accuracy.metric = Metric(self.metric)
        accuracy.dataloader = self.generate_dataloader_config()
        return accuracy

    def generate_performance_config(self) -> Performance:
        """Generate accuracy evaluation configuration."""
        performance = Performance()
        performance.warmup = self.warmup_iterations
        performance.iteration = self.iterations
        performance.configs.num_of_instance = self.num_of_instance
        performance.configs.cores_per_instance = self.cores_per_instance
        performance.dataloader = self.generate_dataloader_config()
        return performance
