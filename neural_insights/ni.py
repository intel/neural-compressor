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

from neural_insights.ux.components.workload_manager.workload_manager import WorkloadManager
from neural_insights.ux.components.workload_manager.workload import Workload
from neural_insights.ux.utils.consts import WorkloadMode


class NeuralInsights:

    def __init__(self, workdir_location: PathLike) -> None:
        self.workdir_location: PathLike = workdir_location

    def add_workload(
            self,
            workload_location: str,
            model_path: str,
            workload_mode: WorkloadMode,
    ) -> None:
        workload = Workload()
        workload.workload_location = workload_location
        workload.mode = workload_mode
        workload.model_path = model_path

        workload_manager = WorkloadManager(workdir_location=self.workdir_location)
        workload_manager.add_workload(workload)

