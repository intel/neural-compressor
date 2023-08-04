# -*- coding: utf-8 -*-
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
"""The workload_manager module contains code used for Neural Insights workloads management."""

import json
import os
import shutil
from os import PathLike
from typing import List, Optional

from neural_insights.components.workload_manager.quantization_workload import \
    QuantizationWorkload, AccuracyData
from neural_insights.components.workload_manager.workload import Workload
from neural_insights.utils.consts import WORKDIR_LOCATION, WorkloadStatus, WorkloadModes
from neural_insights.utils.exceptions import InternalException, ClientErrorException
from neural_insights.utils.json_serializer import JsonSerializer
from neural_insights.utils.logger import log
from neural_insights.utils.singleton import Singleton


class WorkloadManager(JsonSerializer, metaclass=Singleton):
    """Workload Manager class."""

    def __init__(self, workdir_location: Optional[PathLike] = None):
        """Initialize WorkloadManager."""
        super().__init__()
        if workdir_location is None:
            log.debug("Using default workdir location.")
            workdir_location = WORKDIR_LOCATION
        self.config_path = os.path.join(workdir_location, "neural_insights", "config.json")
        self._version: int = 1
        self._workloads: List[Workload] = []

        self.load_workloads()
        self.dump_config()

    def add_workload(self, workload: Workload) -> None:
        """Add workload."""
        self._workloads.append(workload)
        self.dump_config()

    def update_workload_status(self, workload_uuid: str, status: str) -> None:
        """Update workload status."""
        try:
            new_status = WorkloadStatus(status)
        except ValueError:
            raise ClientErrorException("Could not parse status.")
        workload = self.get_workload(workload_uuid)
        current_status = workload.status
        self.validate_status_flow(current_status, new_status)
        workload.status = new_status
        self.dump_config()

    def update_workload_accuracy_data(
            self,
            workload_uuid: str,
            baseline_accuracy: float,
            optimized_accuracy: float,
    ) -> None:
        """Update workload status."""
        workload = self.get_workload(workload_uuid)
        if not isinstance(workload, QuantizationWorkload):
            raise ClientErrorException(
                "Accuracy data is only available in quantization workloads.",
            )

        accuracy_data = AccuracyData(
            {
                "baseline_accuracy": baseline_accuracy,
                "optimized_accuracy": optimized_accuracy,
            },
        )
        workload.accuracy_data = accuracy_data
        self.dump_config()

    @staticmethod
    def validate_status_flow(
            prev_status: WorkloadStatus,
            new_status: WorkloadStatus,
    ) -> None:
        """Validate status flow."""
        status_indices = {
            WorkloadStatus.NEW: 0,
            WorkloadStatus.WIP: 1,
            WorkloadStatus.SUCCESS: 2,
            WorkloadStatus.FAILURE: 3,
        }
        status_connections = [(0, 1), (1, 2), (1, 3)]

        # Initialize adjacency matrix with zeros
        adjacency_matrix = [[0 for _ in status_indices] for _ in status_indices]
        for (source, target) in status_connections:
            adjacency_matrix[source][target] = 1

        prev_status_indice = status_indices[prev_status]
        new_status_indice = status_indices[new_status]
        if adjacency_matrix[prev_status_indice][new_status_indice] != 1:
            raise ClientErrorException(
                f"Changing status from {prev_status.value} to {new_status.value} is not allowed.",
            )

    def load_workloads(self) -> None:
        """Read workloads from config file."""
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path, "r") as config_file:
            config_data = json.load(config_file)
        config_version = int(config_data.get("version"))
        if config_version != self._version:
            raise InternalException(
                f"Incompatible config version has been found. "
                f"Expected version: {self._version}, found {config_version}.",
            )
        workloads_config = config_data.get("workloads", [])
        if not isinstance(workloads_config, list):
            raise InternalException(
                "Incompatible format of workloads. Workloads should be a list.",
            )
        self._workloads = []
        for workload_config in workloads_config:
            if workload_config.get("mode") == WorkloadModes.QUANTIZATION.value:
                workload = QuantizationWorkload(workload_config)
            else:
                workload = Workload(workload_config)
            self._workloads.append(workload)

    def dump_config(self) -> None:
        """Dump workloads to config file."""
        serialized_workloads = [workload.serialize() for workload in self._workloads]
        data = {
            "version": self._version,
            "workloads": serialized_workloads,
        }

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as config_file:
            json.dump(data, config_file)

    @property
    def workloads(self) -> List[Workload]:
        """Get workloads list."""
        self.load_workloads()
        return self._workloads

    def get_workload(self, workload_uuid: str) -> Workload:
        """Get workload from workloads list."""
        for workload in self.workloads:
            if workload.uuid == workload_uuid:
                return workload
        raise ClientErrorException("Could not find workload with specified ID.")

    def remove_workload(self, workload_uuid: str) -> str:
        """Remove workload from workloads list."""
        for index, workload in enumerate(self.workloads):
            if workload.uuid == workload_uuid:
                shutil.rmtree(workload.workload_location, ignore_errors=True)
                del self._workloads[index]
                self.dump_config()
                return workload.uuid
        raise ClientErrorException("Could not find workload with specified ID.")
