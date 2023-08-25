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
"""The workload module for Neural Insights workloads."""

import os
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4

from neural_insights.utils.consts import WorkloadModes, Frameworks, WorkloadStatus
from neural_insights.utils.exceptions import InternalException
from neural_insights.utils.json_serializer import JsonSerializer
from neural_insights.utils.utils import get_framework_from_path


class Workload(JsonSerializer):
    """Workload class."""

    def __init__(self, data: Optional[dict] = None):
        """Initialize Workload."""
        super().__init__()
        if data is None:
            data = {}
        self.uuid = str(data.get("uuid", uuid4()))
        self.creation_time: int = int(data.get("creation_time", datetime.now().timestamp()))
        self.workload_location: str = data.get("workload_location", None)

        mode = data.get("mode")
        self.workload_name = data.get("workload_name", mode)
        if not isinstance(mode, WorkloadModes) and isinstance(mode, str):
            mode = WorkloadModes(mode)
        self.mode: WorkloadModes = mode

        status = data.get("status", WorkloadStatus.NEW)
        if not isinstance(status, WorkloadStatus) and isinstance(status, str):
            status = WorkloadStatus(status)
        self.status: WorkloadStatus = status

        self._model_path: str = data.get("model_path", None)

        framework = data.get("framework", None)
        if not isinstance(framework, Frameworks):
            if framework is None and self.model_path is not None:
                framework = get_framework_from_path(self.model_path)
            if isinstance(framework, str):
                framework = Frameworks(framework)
        self.framework: Optional[Frameworks] = framework

        self.model_summary_file = data.get("model_summary_file", None)

    @property
    def model_path(self) -> str:
        """Get model_path."""
        return self._model_path

    @model_path.setter
    def model_path(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise InternalException("Could not locate model.")
        self._model_path = model_path
        self.framework = Frameworks(get_framework_from_path(self.model_path))

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize Workload class."""
        return {
            "uuid": self.uuid,
            "workload_name": self.workload_name,
            "framework": self.framework.value,
            "workload_location": self.workload_location,
            "mode": self.mode.value,
            "model_path": self.model_path,
            "status": self.status.value,
            "creation_time": self.creation_time,
        }
