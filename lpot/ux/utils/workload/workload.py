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
"""Workload module."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from lpot.ux.components.optimization import Optimizations
from lpot.ux.utils.consts import Precisions
from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.json_serializer import JsonSerializer
from lpot.ux.utils.logger import log
from lpot.ux.utils.utils import (
    get_file_extension,
    get_framework_from_path,
    get_predefined_config_path,
)
from lpot.ux.utils.workload.config import Config


class Workload(JsonSerializer):
    """Workload class."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize Workload class."""
        super().__init__()
        self.config: Config = Config()

        self.id: str = str(data.get("id", ""))
        if not self.id:
            raise ClientErrorException("Workload ID not specified.")

        self.model_path: str = data.get("model_path", "")
        if not self.model_path:
            raise ClientErrorException("Model path is not defined!")

        self.model_name = Path(self.model_path).stem

        self.domain: str = data.get("domain", None)

        if not self.domain:
            raise ClientErrorException("Domain is not defined!")

        self.framework: str = data.get(
            "framework",
            get_framework_from_path(self.model_path),
        )

        self.workspace_path = data.get(
            "workspace_path",
            os.path.dirname(self.model_path),
        )
        self.workload_path = data.get(
            "workload_path",
            os.path.join(
                self.workspace_path,
                "workloads",
                f"{self.model_name}_{self.id}",
            ),
        )

        self.set_workspace()

        self.eval_dataset_path: str = data.get("eval_dataset_path", "")
        self.calib_dataset_path: str = data.get("eval_dataset_path", "")
        self.set_dataset_paths(data)

        for dataset_path in [self.eval_dataset_path, self.calib_dataset_path]:
            if dataset_path != "no_dataset_location" and not os.path.exists(
                dataset_path,
            ):
                raise ClientErrorException(
                    f'Could not found dataset in specified location: "{dataset_path}".',
                )

        if not os.path.isfile(self.model_path):
            raise ClientErrorException(
                f'Could not found model in specified location: "{self.model_path}".',
            )

        self.accuracy_goal: float = data.get("accuracy_goal", 0.01)

        self.config_name = "config.yaml"
        self.predefined_config_path = data.get(
            "config_path",
            get_predefined_config_path(self.framework, self.domain),
        )
        self.config_path = os.path.join(
            self.workload_path,
            self.config_name,
        )

        self.input_precision = Precisions.FP32  # TODO: Detect input model precision
        self.output_precision = data.get("precision", data.get("output_precision"))

        self.mode = self.get_optimization_mode()
        self.tune = data.get("tune", self.is_tuning_enabled(data))

        self.initialize_config(data)

        self.input_nodes: Optional[str] = data.get("inputs", data.get("input_nodes"))
        self.output_nodes: Optional[str] = data.get("outputs", data.get("output_nodes"))

        self.model_output_path = os.path.join(
            self.workload_path,
            self.model_output_name,
        )

    def initialize_config(self, data: dict) -> None:
        """Initialize config."""
        if not os.path.isfile(self.config_path):
            self.config.load(self.predefined_config_path)
        else:
            self.config.load(self.config_path)

        config_initializers = {
            Optimizations.TUNING: self.initialize_tuning_config,
            Optimizations.GRAPH: self.initialize_graph_optimization_config,
        }
        initializer = config_initializers.get(self.mode, None)
        if initializer is None:
            raise ClientErrorException(f"Could not initialize config for {self.mode} mode.")
        self.config.model.name = self.model_name
        self.config.set_evaluation_dataset_path(self.eval_dataset_path)
        self.config.set_workspace(self.workload_path)
        self.config.set_accuracy_goal(self.accuracy_goal)
        initializer()

    def initialize_tuning_config(self) -> None:
        """Initialize tuning config."""
        self.config.set_quantization_dataset_path(self.calib_dataset_path)

    def initialize_graph_optimization_config(self) -> None:
        """Initialize graph optimization config."""
        self.config.quantization = None
        self.config.pruning = None
        self.config.set_optimization_precision(self.output_precision)

    def get_optimization_mode(self) -> str:
        """Get optimization mode based on precision."""
        modes_map = {
            Precisions.INT8: Optimizations.TUNING,
            Precisions.FP32: Optimizations.GRAPH,
            Precisions.MIXED: Optimizations.GRAPH,
        }
        mode = modes_map.get(self.output_precision, None)
        if mode is None:
            raise ClientErrorException(
                f"Could not found optimization mode for {self.output_precision} precision.",
            )
        return mode

    def set_dataset_paths(self, data: dict) -> None:
        """Set calibration and evaluation dataset path."""
        if data.get("evaluation", {}).get("dataset_path"):
            self.eval_dataset_path = data.get("evaluation", {}).get("dataset_path")
        if data.get("quantization", {}).get("dataset_path"):
            self.calib_dataset_path = data.get("quantization", {}).get("dataset_path")

        if not self.eval_dataset_path:
            self.eval_dataset_path = data.get("dataset_path", "")
        if not self.calib_dataset_path:
            self.calib_dataset_path = data.get("dataset_path", "")

    def set_workspace(self) -> None:
        """Create (if missing) necessary folders for workloads."""
        os.makedirs(self.workspace_path, exist_ok=True)
        os.makedirs(self.workload_path, exist_ok=True)

    def dump(self) -> None:
        """Dump workload to yaml."""
        json_path = os.path.join(self.workload_path, "workload.json")
        with open(json_path, "w") as f:
            json.dump(self.serialize(), f, indent=4)

        log.debug(f"Successfully saved workload to {json_path}")

    def is_tuning_enabled(self, data: dict) -> bool:
        """Check if tuning is enabled for workload."""
        if self.output_precision == Precisions.INT8:
            return True
        elif self.output_precision == Precisions.FP32:
            return False
        elif self.output_precision == Precisions.MIXED:
            if data.get("tuning"):
                return True
            else:
                return False
        else:
            raise ClientErrorException(f"Precision {self.output_precision} is not supported.")

    @property
    def model_output_name(self) -> str:
        """Get output model name."""
        output_name = self.model_name
        if self.mode == Optimizations.TUNING:
            output_name += "_tuned_" + self.output_precision
        elif self.mode == Optimizations.GRAPH:
            output_name = self.model_name + "_optimized_"
            if self.tune:
                output_name += "tuned_"
            output_name += "_".join(
                [precision.strip() for precision in self.output_precision.split(",")],
            )
        else:
            raise ClientErrorException(f"Mode {self.mode} is not supported.")

        return output_name + "." + get_file_extension(self.model_path)
