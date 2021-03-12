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
"""Tuning class."""

import os
import re
from typing import Any, Dict, Optional

from lpot.ux.utils.hw_info import HWInfo
from lpot.ux.utils.workload.workload import Workload


class Tuning:
    """Tuning class."""

    def __init__(
        self,
        workload: Workload,
        workload_path: str,
        template_path: Optional[str] = None,
    ) -> None:
        """Initialize configuration Dataset class."""
        self.model_path = workload.model_path
        self.framework = workload.framework

        self.instances: int = 1
        self.cores_per_instance: int = HWInfo().cores // self.instances

        self.model_output_path = workload.model_output_path
        self.config_path = workload.config_path

        self.script_path = os.path.join(os.path.dirname(__file__), "tune_model.py")

        self.command = [
            "python",
            self.script_path,
            "--input-graph",
            self.model_path,
            "--output-graph",
            self.model_output_path,
            "--config",
            self.config_path,
            "--framework",
            self.framework,
        ]

        if template_path:
            self.command = ["python", template_path]

    def serialize(self) -> Dict[str, Any]:
        """Serialize Tuning to dict."""
        result = {}
        for key, value in self.__dict__.items():
            variable_name = re.sub(r"^_", "", key)
            if variable_name == "command":
                result[variable_name] = " ".join(value)
            else:
                result[variable_name] = value
        return result
