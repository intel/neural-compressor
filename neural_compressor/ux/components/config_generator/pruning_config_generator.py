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
"""Pruning configuration generator class."""
from typing import Any

from neural_compressor.ux.components.config_generator.config_generator import ConfigGenerator
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.utils.workload.evaluation import Accuracy, Evaluation, Metric
from neural_compressor.ux.utils.workload.pruning import Pruning


class PruningConfigGenerator(ConfigGenerator):
    """PruningConfigGenerator class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize configuration generator."""
        super().__init__(*args, **kwargs)
        data = kwargs.get("data", {})
        self.pruning_configuration: dict = data["pruning_details"]

    def generate(self) -> None:
        """Generate yaml config file."""
        config = Config()
        config.load(self.predefined_config_path)
        config.quantization = None
        config.model = self.generate_model_config()
        config.evaluation = self.generate_evaluation_config()
        config.pruning = self.generate_pruning_config()
        config.dump(self.config_path)

    def generate_evaluation_config(self) -> Evaluation:
        """Generate evaluation configuration."""
        evaluation = Evaluation()
        evaluation.accuracy = Accuracy()

        if self.metric:
            evaluation.accuracy.metric = Metric(self.metric)

        evaluation.accuracy.dataloader = self.generate_dataloader_config(batch_size=1)
        evaluation.set_accuracy_postprocess_transforms(self.transforms)
        return evaluation

    def generate_pruning_config(self) -> Pruning:
        """Generate graph optimization configuration."""
        pruning = Pruning(self.pruning_configuration)
        if pruning.train is not None:
            pruning.train.dataloader = self.generate_dataloader_config(batch_size=1)
            pruning.train.set_postprocess_transforms(self.transforms)
        return pruning
