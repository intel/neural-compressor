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

"""Workload service."""
from neural_compressor.ux.components.optimization.tuning_history import tuning_history
from neural_compressor.ux.utils.exceptions import NotFoundException
from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.web.communication import MessageQueue
from neural_compressor.ux.web.service.request_data_processor import RequestDataProcessor
from neural_compressor.ux.web.service.response_generator import Response, ResponseGenerator

mq = MessageQueue()


class WorkloadService:
    """Workload related services."""

    @staticmethod
    def get_config(data: dict) -> Response:
        """Get config file for requested Workload."""
        config_path = WorkloadService._get_workload_data_from_input_data(data).get("config_path")

        if not config_path:
            raise NotFoundException("Unable to find config file")

        return ResponseGenerator.serve_from_filesystem(
            path=config_path,
            mimetype="text/vnd.yaml",
        )

    @staticmethod
    def get_code_template(data: dict) -> Response:
        """Get code template file for requested Workload."""
        code_template_path = WorkloadService._get_workload_data_from_input_data(data).get(
            "code_template_path",
        )

        if not code_template_path:
            raise NotFoundException("Unable to find code template file")

        return ResponseGenerator.serve_from_filesystem(
            path=code_template_path,
            mimetype="text/x-python",
        )

    @staticmethod
    def get_output(data: dict) -> Response:
        """Get config file for requested Workload."""
        log_path = WorkloadService._get_workload_data_from_input_data(data).get("log_path")

        if not log_path:
            raise NotFoundException("Unable to find output log")

        try:
            response = ResponseGenerator.serve_from_filesystem(
                path=log_path,
                mimetype="text/plain",
            )
        except Exception as e:
            response = ResponseGenerator.from_exception(e)

        return ResponseGenerator.add_refresh(
            response=response,
            refresh_time=3,
        )

    @staticmethod
    def request_history_snapshot(data: dict) -> None:
        """Get tuning history for requested Workload."""
        workload_id = RequestDataProcessor.get_string_value(data, "workload_id")
        WorkloadService.send_history_snapshot(workload_id)

    @staticmethod
    def send_history_snapshot(workload_id: str) -> None:
        """Get tuning history for requested Workload."""
        try:
            response = tuning_history(workload_id)
            mq.post_success("tuning_history", response)
        except NotFoundException:
            mq.post_error(
                "tuning_history",
                {
                    "workload_id": workload_id,
                },
            )

    @staticmethod
    def _get_workload_data_from_input_data(data: dict) -> dict:
        """Return data for requested Workload."""
        workload_id = RequestDataProcessor.get_string_value(data, "workload_id")
        return WorkloadService.get_workload_data_by_id(workload_id)

    @staticmethod
    def get_workload_data_by_id(workload_id: str) -> dict:
        """Return data for requested Workload."""
        workdir = Workdir()
        workload_data = workdir.get_workload_data(workload_id)
        if not workload_data:
            raise NotFoundException(f"Unable to find workload with id: {workload_id}")
        return workload_data
