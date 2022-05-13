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
import os
from typing import List

from neural_compressor.ux.components.db_manager.db_operations import (
    ProfilingAPIInterface,
    ProjectAPIInterface,
)
from neural_compressor.ux.components.profiling.profiling import Profiling
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.utils import export_to_csv
from neural_compressor.ux.web.communication import MessageQueue
from neural_compressor.ux.web.service.request_data_processor import RequestDataProcessor
from neural_compressor.ux.web.service.response_generator import Response, ResponseGenerator
from neural_compressor.ux.web.service.workload import WorkloadService

mq = MessageQueue()


class ProfilingService(WorkloadService):
    """Profiling related services."""

    @classmethod
    def generate_csv(cls, data: dict) -> Response:
        """Get config file for requested Workload."""
        profiling_data = cls._get_workload_data(data)

        profiling_results = profiling_data.get("results", None)
        if profiling_results is None:
            raise ClientErrorException("Could not find results for specified project.")

        project_id = profiling_data["project_id"]
        project_details = ProjectAPIInterface.get_project_details({"id": project_id})

        processed_results = cls._process_results_for_export(profiling_results)
        profiling = Profiling(project_data=project_details, profiling_data=profiling_data)

        csv_file_path = os.path.join(profiling.workdir, "results.csv")
        export_to_csv(data=processed_results, file=csv_file_path)

        return ResponseGenerator.serve_from_filesystem(
            path=csv_file_path,
            as_attachment=True,
        )

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

    @classmethod
    def _process_results_for_export(cls, results: List[dict]) -> List[dict]:
        """Process results by removing not important data."""
        processed_results: List[dict] = []
        keys_to_remove = ["profiling_id", "id", "created_at"]
        for result in results:
            processed_results.append(
                {key: val for key, val in result.items() if key not in keys_to_remove},
            )
        return processed_results
