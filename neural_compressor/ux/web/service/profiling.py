# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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

"""Profiling service."""
from neural_compressor.ux.components.db_manager.db_operations import ProfilingAPIInterface
from neural_compressor.ux.web.service.request_data_processor import RequestDataProcessor
from neural_compressor.ux.web.service.workload import WorkloadService


class ProfilingService(WorkloadService):
    """Profiling related services."""

    @staticmethod
    def _get_workload_data(data: dict) -> dict:
        """Return data for requested Workload."""
        profiling_id = RequestDataProcessor.get_string_value(data, "id")
        profiling_data = ProfilingAPIInterface.get_profiling_details(
            {
                "id": profiling_id,
            },
        )
        return profiling_data
