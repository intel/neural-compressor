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

from lpot.ux.utils.json_serializer import JsonSerializer
from lpot.ux.utils.logger import log
from lpot.ux.utils.utils import (
    get_framework_from_path,
    get_model_domain,
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
            raise Exception("Workload ID not specified.")

        self.model_path: str = data.get("model_path", "")
        if not self.model_path:
            raise Exception("Model path is not defined!")

        self.model_name = Path(self.model_path).stem
        self.workspace_path: str = os.path.join(
            data.get("workspace_path", os.path.dirname(self.model_path)),
            self.id,
        )
        self.set_workspace()

        self.config_name = f"config.{self.id}.yaml"
        self.config_path = data.get("config_path", None)
        if not self.config_path:
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

    def set_workspace(self) -> None:
        """Update workspace path in LPOT config."""
        os.makedirs(self.workspace_path, exist_ok=True)

    def dump(self) -> None:
        """Dump workload to yaml."""
        json_path = os.path.join(self.workspace_path, f"workload.{self.id}.json")
        with open(json_path, "w") as f:
            json.dump(self.serialize(), f, indent=4)

        log.debug(f"Successfully saved workload to {json_path}")
