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

"""Workload service."""
from neural_compressor.ux.utils.exceptions import InternalException, NotFoundException
from neural_compressor.ux.web.communication import MessageQueue
from neural_compressor.ux.web.service.response_generator import Response, ResponseGenerator

mq = MessageQueue()


class WorkloadService:
    """Workload related services."""

    @classmethod
    def get_config(cls, data: dict) -> Response:
        """Get config file for requested Workload."""
        config_path = cls._get_workload_data(data).get("config_path")

        if not config_path:
            raise NotFoundException("Unable to find config file")

        return ResponseGenerator.serve_from_filesystem(
            path=config_path,
            mimetype="text/vnd.yaml",
        )

    @classmethod
    def get_code_template(cls, data: dict) -> Response:
        """Get code template file for requested Workload."""
        code_template_path = cls._get_workload_data(data).get(
            "code_template_path",
        )

        if not code_template_path:
            raise NotFoundException("Unable to find code template file")

        return ResponseGenerator.serve_from_filesystem(
            path=code_template_path,
            mimetype="text/x-python",
        )

    @classmethod
    def get_output(cls, data: dict) -> Response:
        """Get config file for requested Workload."""
        log_path = cls._get_workload_data(data).get("log_path")

        if not log_path:
            raise NotFoundException("Unable to find output log")

        refresh_time = int(data.get("autorefresh", [0])[0])
        try:
            response = ResponseGenerator.serve_from_filesystem(
                path=log_path,
                mimetype="text/plain",
            )
        except Exception as e:
            response = ResponseGenerator.from_exception(e)

        if refresh_time > 0:
            response = ResponseGenerator.add_refresh(
                response=response,
                refresh_time=refresh_time,
            )
        return response

    @staticmethod
    def _get_workload_data(data: dict) -> dict:
        """Return data for requested Workload."""
        raise InternalException("Not implemented workload.")
