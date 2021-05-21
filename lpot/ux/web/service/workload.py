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

from lpot.ux.utils.exceptions import NotFoundException
from lpot.ux.utils.templates.workdir import Workdir
from lpot.ux.web.service.request_data_processor import RequestDataProcessor
from lpot.ux.web.service.response_generator import Response, ResponseGenerator


class WorkloadService:
    """Workload related services."""

    @staticmethod
    def get_config(data: dict) -> Response:
        """Get config file for requested Workload."""
        workload_id = RequestDataProcessor.get_string_value(data, "workload_id")
        workdir = Workdir()
        workload_data = workdir.get_workload_data(workload_id)
        config_path = workload_data.get("config_path")

        if not config_path:
            raise NotFoundException(f"Unable to find config file for {workload_id}")

        return ResponseGenerator.serve_from_filesystem(
            path=config_path,
            mimetype="text/vnd.yaml",
        )

    @staticmethod
    def get_code_template(data: dict) -> Response:
        """Get code template file for requested Workload."""
        workload_id = RequestDataProcessor.get_string_value(data, "workload_id")
        workdir = Workdir()
        workload_data = workdir.get_workload_data(workload_id)
        code_template_path = workload_data.get("code_template_path")

        if not code_template_path:
            raise NotFoundException(f"Unable to find code template file for {workload_id}")

        return ResponseGenerator.serve_from_filesystem(
            path=code_template_path,
            mimetype="text/x-python",
        )

    @staticmethod
    def get_output(data: dict) -> Response:
        """Get config file for requested Workload."""
        workload_id = RequestDataProcessor.get_string_value(data, "workload_id")
        workdir = Workdir()
        workload_data = workdir.get_workload_data(workload_id)
        log_path = workload_data.get("log_path")

        if not log_path:
            raise NotFoundException(f"Unable to find output log for {workload_id}")

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
