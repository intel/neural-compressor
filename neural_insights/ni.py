# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
"""Neural Insights main class."""
from os import PathLike
from typing import Optional

from neural_insights.components.workload_manager.quantization_workload import QuantizationWorkload
from neural_insights.components.workload_manager.workload import Workload
from neural_insights.components.workload_manager.workload_manager import WorkloadManager
from neural_insights.utils.consts import WorkloadModes


class NeuralInsights:
    def __init__(self, workdir_location: PathLike) -> None:
        self.workdir_location: PathLike = workdir_location

    def add_workload(
        self,
        workload_location: str,
        model_path: str,
        workload_mode: WorkloadModes,
            workload_name: str,
            model_summary_file: Optional[str]
    ) -> str:
        """Add workload to Neural Insights."""
        if workload_mode == WorkloadModes.QUANTIZATION:
            workload = QuantizationWorkload()
        else:
            workload = Workload()
        workload.workload_name = workload_name
        workload.workload_location = workload_location
        workload.mode = workload_mode
        workload.model_path = model_path
        workload.model_summary_file = model_summary_file

        workload_manager = WorkloadManager(workdir_location=self.workdir_location)
        workload_manager.add_workload(workload)

        return workload.uuid

    def update_workload_status(self, workload_uuid: str, status: str) -> None:
        """Update status of specified workload."""
        workload_manager = WorkloadManager(workdir_location=self.workdir_location)
        workload_manager.update_workload_status(workload_uuid, status)

    def update_workload_accuracy_data(
        self,
        workload_uuid: str,
        baseline_accuracy: float,
        optimized_accuracy: float,
    ) -> None:
        """Update accuracy data of specified workload."""
        workload_manager = WorkloadManager(workdir_location=self.workdir_location)
        workload_manager.update_workload_accuracy_data(
            workload_uuid,
            baseline_accuracy,
            optimized_accuracy,
        )
