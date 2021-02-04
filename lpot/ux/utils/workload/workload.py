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
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from lpot.ux.utils.json_serializer import JsonSerializer
from lpot.ux.utils.utils import (
    find_boundary_nodes,
    get_framework_from_path,
    get_model_domain,
    get_predefined_config_path,
)
from lpot.ux.utils.workload.config import Config

logging.basicConfig(level=logging.INFO)


class Workload(JsonSerializer):
    """Workload class."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize Workload class."""
        super().__init__()
        self.config: Config = Config()

        self.id: Optional[str] = data.get("id")
        if self.id is None:
            raise Exception("Workload ID not specified.")

        self.model_path: str = data.get("model_path", "")
        if not self.model_path:
            raise Exception("Model path is not defined!")

        self.model_name = Path(self.model_path).stem
        self.workspace_path: str = data.get(
            "workspace_path",
            os.path.dirname(self.model_path),
        )

        self.config_path = os.path.join(
            self.workspace_path,
            f"config.{self.id}.yaml",
        )

        self.framework: str = data.get(
            "framework",
            get_framework_from_path(self.model_path),
        )

        self.domain: str = data.get("domain", "")
        if not self.domain:
            self.domain = get_model_domain(self.model_name)

        self.dataset_path: str = data.get("dataset_path", "")

        if not os.path.isdir(self.dataset_path):
            raise Exception(
                f'Could not found dataset in specified location: "{self.dataset_path}".',
            )

        if not os.path.isfile(self.model_path):
            raise Exception(
                f'Could not found model in specified location: "{self.model_path}".',
            )

        self.accuracy_goal: float = data.get("accuracy_goal", 0.01)

        if not os.path.isfile(self.config_path):
            self.config.load(get_predefined_config_path(self.framework, self.domain))
        else:
            self.config.load(self.config_path)

        self.config.model.name = self.model_name
        self.config.set_dataset_path(self.dataset_path)
        self.config.set_workspace(self.workspace_path)
        self.config.set_accuracy_goal(self.accuracy_goal)
        self.set_boundary_nodes(data)

    def set_boundary_nodes(self, data: Dict[str, Any]) -> None:
        """Set model input and output nodes."""
        detected_nodes = {}
        if not data.get("input") or not data.get("output") is None:
            detected_nodes = find_boundary_nodes(self.model_path)

        if data.get("input"):
            input_nodes = data.get("input")
        else:
            input_nodes = detected_nodes.get("inputs", [])

        if data.get("output"):
            output_nodes = data.get("output")
        else:
            output_nodes = detected_nodes.get("outputs", [])

        self.config.model.inputs = input_nodes  # type: ignore
        self.config.model.outputs = output_nodes  # type: ignore

    def dump(self) -> None:
        """Dump workload to yaml."""
        json_path = os.path.join(self.workspace_path, f"workload.{self.id}.json")
        with open(json_path, "w") as f:
            json.dump(self.serialize(), f, indent=4)

        logging.info(f"Successfully saved workload to {json_path}")
