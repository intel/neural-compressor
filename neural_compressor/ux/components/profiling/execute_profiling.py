# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022 Intel Corporation
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

"""Execute profiling."""

import os
from typing import Any, Dict

from neural_compressor.ux.components.db_manager.db_operations import (
    ProfilingAPIInterface,
    ProjectAPIInterface,
)
from neural_compressor.ux.components.profiling.profiling import Profiling
from neural_compressor.ux.utils.consts import ExecutionStatus
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.executor import Executor
from neural_compressor.ux.utils.parser import ProfilingParser
from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.web.communication import MessageQueue

mq = MessageQueue()


def execute_profiling(data: Dict[str, Any]) -> None:
    """
    Execute profiling.

    Expected data:
    {
        "request_id": "asd",
        "profiling_id": "1"
    }
    """
    if not all([str(data.get("request_id", "")), str(data.get("profiling_id", ""))]):
        message = "Missing request id or profiling id."
        mq.post_error(
            "profiling_finish",
            {"message": "Failure", "code": 404},
        )
        raise ClientErrorException(message)

    request_id: str = str(data["request_id"])
    profiling_id: int = int(data["profiling_id"])

    try:
        profiling_details = ProfilingAPIInterface.get_profiling_details({"id": profiling_id})

        project_id = profiling_details["project_id"]
        project_details = ProjectAPIInterface.get_project_details({"id": project_id})

        response_data = execute_real_profiling(
            request_id=request_id,
            project_details=project_details,
            profiling_details=profiling_details,
        )

        mq.post_success("profiling_finish", response_data)
    except Exception:
        ProfilingAPIInterface.update_profiling_status(
            {
                "id": profiling_id,
                "status": ExecutionStatus.ERROR,
            },
        )
        mq.post_error(
            "profiling_finish",
            {"message": "Failure", "code": 404, "request_id": request_id},
        )
        raise


def execute_real_profiling(
    request_id: str,
    project_details: dict,
    profiling_details: dict,
) -> dict:
    """Execute profiling."""
    profiling: Profiling = Profiling(project_details, profiling_details)

    Workdir.clean_logs(profiling.workdir)
    profiling.generate_config()

    logs = [os.path.join(profiling.workdir, "output.txt")]

    ProfilingAPIInterface.update_log_path(
        {
            "id": profiling.profiling_id,
            "log_path": logs[0],
        },
    )

    ProfilingAPIInterface.update_execution_command(
        {
            "id": profiling.profiling_id,
            "execution_command": profiling.command,
        },
    )

    send_data = {
        "message": "started",
        "request_id": request_id,
        "output_path": logs[0],
    }

    executor = Executor(
        workspace_path=profiling.workdir,
        subject="profiling",
        data=send_data,
        log_name="output",
    )

    proc = executor.call(
        profiling.command,
    )

    parser = ProfilingParser(logs)
    parsed_data = parser.process()

    if not proc.is_ok:
        raise InternalException("Profiling failed during execution.")

    ProfilingAPIInterface.bulk_add_results(
        profiling_id=profiling.profiling_id,
        results=parsed_data.get("profiling_data", []),
    )

    ProfilingAPIInterface.update_profiling_duration(
        {
            "id": profiling.profiling_id,
            "duration": executor.process_duration,
        },
    )

    ProfilingAPIInterface.update_profiling_status(
        {
            "id": profiling.profiling_id,
            "status": ExecutionStatus.SUCCESS,
        },
    )

    profiling_data = ProfilingAPIInterface.get_profiling_details(
        {
            "id": profiling.profiling_id,
        },
    )

    response_data = {
        "request_id": request_id,
    }

    response_data.update(profiling_data)

    return response_data
