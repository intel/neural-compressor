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
"""Benchmark class."""

import os

from lpot.ux.components.lpot_configuration_wizard.params_feeder import load_model_config
from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.workload.workload import Workload


class Benchmark:
    """LPOT Benchmark class."""

    def __init__(self, workload: Workload, model_path: str) -> None:
        """Initialize LPOT Benchmark class."""
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise ClientErrorException("Could not found model in specified path.")

        self.config_path = workload.config_path

        model_config = load_model_config()
        script_relative_path = (
            model_config.get(workload.framework, {})
            .get(workload.domain, {})
            .get(workload.model_name, {})
            .get("script_path", None)
        )
        self.script_path = os.path.join(
            os.environ["LPOT_REPOSITORY_PATH"],
            script_relative_path,
        )
        self.command = [
            "python",
            self.script_path,
            "--input-graph",
            self.model_path,
            "--config",
            self.config_path,
            "--benchmark",
        ]
