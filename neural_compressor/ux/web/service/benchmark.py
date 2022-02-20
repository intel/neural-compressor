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

"""Benchmark service."""
from neural_compressor.ux.components.db_manager.db_operations import BenchmarkAPIInterface
from neural_compressor.ux.web.service.request_data_processor import RequestDataProcessor
from neural_compressor.ux.web.service.workload import WorkloadService


class BenchmarkService(WorkloadService):
    """Benchmark related services."""

    @staticmethod
    def _get_workload_data(data: dict) -> dict:
        """Return data for requested Workload."""
        benchmark_id = RequestDataProcessor.get_string_value(data, "id")
        benchmark_data = BenchmarkAPIInterface.get_benchmark_details(
            {
                "id": benchmark_id,
            },
        )
        return benchmark_data
