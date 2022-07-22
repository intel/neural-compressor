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
from typing import Any, Optional

from neural_compressor.ux.components.config_generator.config_generator import ConfigGenerator
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.utils.workload.evaluation import Accuracy, Evaluation, Metric
from neural_compressor.ux.utils.workload.quantization import Calibration, Quantization, WiseConfig
from neural_compressor.ux.utils.workload.tuning import (
    AccCriterion,
    Diagnosis,
    ExitPolicy,
    MultiObjectives,
    Strategy,
    Tuning,
)


class QuantizationConfigGenerator(ConfigGenerator):
    """Configuration generator class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize configuration generator."""
        super().__init__(*args, **kwargs)
        data = kwargs.get("data", {})
        self.sampling_size: int = data["sampling_size"]
        self.tuning_strategy: dict = data["tuning_strategy"]
        self.accuracy_criterion: dict = data["accuracy_criterion"]
        self.objective: str = data["objective"]
        self.exit_policy: dict = data["exit_policy"]
        self.random_seed: int = data["random_seed"]
        diagnosis_config: Optional[dict] = data.get("diagnosis_config", {})
        if diagnosis_config is None:
            diagnosis_config = {}

        self.op_wise: Optional[dict] = diagnosis_config.get("op_wise", None)
        self.model_wise: Optional[WiseConfig] = WiseConfig(
            diagnosis_config.get("model_wise", None),
        )

    def generate(self) -> None:
        """Generate yaml config file."""
        config = Config()
        config.load(self.predefined_config_path)
        config.model = self.generate_model_config()
        config.tuning = self.generate_tuning_config()
        config.quantization = self.generate_quantization_config()
        config.evaluation = self.generate_evaluation_config()
        config.dump(self.config_path)

    def generate_tuning_config(self) -> Tuning:
        """Generate tuning configuration."""
        tuning = Tuning()
        tuning.strategy = Strategy({"name": self.tuning_strategy})
        tuning.accuracy_criterion = AccCriterion(self.accuracy_criterion)
        tuning.multi_objectives = MultiObjectives({"objective": self.objective})
        tuning.exit_policy = ExitPolicy(self.exit_policy)
        tuning.random_seed = self.random_seed

        tuning.diagnosis = Diagnosis()

        tuning.set_workspace(self.workdir)
        return tuning

    def generate_quantization_config(self) -> Quantization:
        """Generate quantization configuration."""
        quantization = Quantization()
        quantization.calibration = Calibration()
        quantization.calibration.sampling_size = self.sampling_size
        if self.dataset_type != "custom":
            quantization.calibration.dataloader = self.generate_dataloader_config()

        if self.op_wise:
            quantization.op_wise = self.op_wise

        if self.model_wise:
            quantization.model_wise = self.model_wise

        return quantization

    def generate_evaluation_config(self) -> Evaluation:
        """Generate evaluation configuration."""
        evaluation = Evaluation()
        evaluation.accuracy = Accuracy()

        if self.metric and self.metric.get("name") != "custom":
            evaluation.accuracy.metric = Metric(self.metric)

        if self.dataset_type != "custom":
            evaluation.accuracy.dataloader = self.generate_dataloader_config(batch_size=1)
            evaluation.set_accuracy_postprocess_transforms(self.transforms)
        return evaluation
