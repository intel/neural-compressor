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
from typing import Any, Dict

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
        self.predefined_config_path = data.get(
            "config_path",
            get_predefined_config_path(self.framework, self.domain),
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
        self.config_name = "config.yaml"
        self.config_path = os.path.join(
            self.workload_path,
            self.config_name,
        )

        model_output_name = (
            self.model_name + "_int8." + get_file_extension(self.model_path)
        )
        self.model_output_path = os.path.join(
            self.workload_path,
            model_output_name,
        )

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

        if not os.path.isfile(self.config_path):
            self.config.load(self.predefined_config_path)
        else:
            self.config.load(self.config_path)

        self.config.model.name = self.model_name
        self.config.set_evaluation_dataset_path(self.eval_dataset_path)
        self.config.set_quantization_dataset_path(self.calib_dataset_path)
        self.config.set_workspace(self.workload_path)
        self.config.set_accuracy_goal(self.accuracy_goal)

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
