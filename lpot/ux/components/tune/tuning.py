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

from lpot.ux.utils.workload.workload import Workload


class Tuning:
    """Tuning class."""

    def __init__(self, workload: Workload, workload_path: str) -> None:
        """Initialize configuration Dataset class."""
        model_output_name = workload.model_name + "_int8.pb"
        self.model_output_path = os.path.join(
            workload_path,
            model_output_name,
        )
        self.model_path = workload.model_path
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
        ]
