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

from lpot.ux.utils.utils import load_model_config
from lpot.ux.utils.workload.workload import Workload


class Tuning:
    """Tuning class."""

    def __init__(self, workload: Workload) -> None:
        """Initialize configuration Dataset class."""
        model_output_name = workload.model_name + "_int8.pb"
        self.model_output_path = os.path.join(
            workload.workspace_path,
            model_output_name,
        )
        self.model_path = workload.model_path
        self.config_path = workload.config_path

        model_config = load_model_config()
        script_path = (
            model_config.get(workload.framework, None)
            .get(workload.domain, None)
            .get(workload.model_name, None)
            .get("script_path", None)
        )
        self.script_path = os.path.join(os.environ["LPOT_REPOSITORY_PATH"], script_path)
        self.command = [
            "python",
            self.script_path,
            "--input-graph",
            self.model_path,
            "--output-graph",
            self.model_output_path,
            "--config",
            self.config_path,
            "--tune",
        ]
