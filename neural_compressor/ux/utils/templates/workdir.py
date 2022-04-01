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
"""Workdir class."""

import glob
import os

from neural_compressor.ux.components.db_manager.db_operations import (
    BenchmarkAPIInterface,
    OptimizationAPIInterface,
    ProfilingAPIInterface,
)
from neural_compressor.ux.utils.consts import WORKDIR_LOCATION, WORKSPACE_LOCATION, ExecutionStatus


class Workdir:
    """Class responsible for workspace management."""

    def __init__(self) -> None:
        """Initialize workdir class."""
        self.workspace_path = WORKSPACE_LOCATION
        self.workdir_path = WORKDIR_LOCATION
        self.ensure_working_path_exists()

    def ensure_working_path_exists(self) -> None:
        """Set workdir if doesn't exist create."""
        os.makedirs(self.workdir_path, exist_ok=True)

    @staticmethod
    def clean_logs(workdir_path: str) -> None:
        """Clean log files."""
        log_files = [os.path.join(workdir_path, "output.txt")]
        log_files.extend(
            glob.glob(
                os.path.join(workdir_path, "*.proc"),
            ),
        )
        log_files.extend(
            glob.glob(
                os.path.join(workdir_path, "*performance_benchmark.txt"),
            ),
        )
        for file in log_files:
            if os.path.exists(file):
                os.remove(file)

    @staticmethod
    def clean_status(status_to_clean: ExecutionStatus) -> None:
        """Clean statuses in database for workloads according to passed parameters."""
        OptimizationAPIInterface.clean_status(status_to_clean)
        BenchmarkAPIInterface.clean_status(status_to_clean)
        ProfilingAPIInterface.clean_status(status_to_clean)
