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

"""Execute profiling."""

import os
from typing import Any, Dict

from neural_compressor.ux.components.profiling.profiling import Profiling
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.executor import Executor
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.parser import ProfilingParser
from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.utils.utils import _load_json_as_dict
from neural_compressor.ux.utils.workload.workload import Workload
from neural_compressor.ux.web.communication import MessageQueue

mq = MessageQueue()


def execute_profiling(data: Dict[str, Any]) -> None:
    """
    Execute profiling setting correct workload status.

    Expected data:
    {
        "id": "configuration_id",
        "model_path": "/localdisk/int8.pb",
        "model_type": <"input_model" | "optimized_model">
    }
    """
    request_id = str(data.get("id", ""))

    if not request_id:
        message = "Missing request id."
        mq.post_error(
            "profiling_finish",
            {"message": "Failure", "code": 404},
        )
        log.error(f"Problem during profiling: {message}")
        return

    workdir = Workdir(request_id=request_id, overwrite=False)
    workdir.set_workload_status(request_id, "wip")

    try:
        execute_real_profiling(
            request_id=request_id,
            workdir=workdir,
            data=data,
        )
    except Exception:
        workdir.set_workload_status(request_id, "error")
        mq.post_error(
            "profiling_finish",
            {"message": "Failure", "code": 404, "id": request_id},
        )
        log.exception("Exception during profiling")
        return

    workdir.set_workload_status(request_id, "success")


def execute_real_profiling(
    request_id: str,
    workdir: Workdir,
    data: Dict[str, Any],
) -> None:
    """
    Execute profiling.

    Expected data:
    {
        "id": "configuration_id",
        "model_path": "/localdisk/int8.pb",
        "model_type": <"input_model" | "optimized_model">
    }
    """
    from neural_compressor.ux.utils.workload.workload import Workload

    model_path = data.get("model_path", None)
    model_type = data.get("model_type", None)

    if not (model_path and model_type):
        message = "Missing model path or type."
        raise ClientErrorException(message)

    workload_path = workdir.workload_path
    workload_data = _load_json_as_dict(
        os.path.join(workload_path, "workload.json"),
    )

    workload = Workload(workload_data)

    response_data: Dict[str, Any] = {"id": request_id, "execution_details": {}}

    mq.post_success(
        "profiling_start",
        {
            "message": "started",
            "id": request_id,
        },
    )
    profile_model_and_respond_to_ui(
        response_data=response_data,
        workload=workload,
        workdir=workdir,
        model=model_type,
        model_path=model_path,
    )
    workdir.set_workload_status(request_id, "success")


def profile_model_and_respond_to_ui(
    response_data: dict,
    workload: Workload,
    workdir: Workdir,
    model: str,
    model_path: str,
) -> dict:
    """Profile model and update UI."""
    response_data = profile_model(
        response_data=response_data,
        workload=workload,
        workdir=workdir,
        model=model,
        model_path=model_path,
    )
    mq.post_success("profiling_finish", response_data)
    return response_data


def profile_model(
    response_data: dict,
    workload: Workload,
    workdir: Workdir,
    model: str,
    model_path: str,
) -> dict:
    """Profile model and prepare response data."""
    request_id = response_data.get("id")

    profiling: Profiling = Profiling(
        workload=workload,
        model_path=model_path,
    )

    log_name = f"{model}_profiling"

    executor = Executor(
        workload.workload_path,
        subject="profiling",
        data={"id": request_id},
        send_response=False,
        log_name=log_name,
        additional_log_names=["output.txt"],
    )

    proc = executor.call(
        profiling.command,
    )

    logs = [os.path.join(workload.workload_path, f"{log_name}.txt")]

    parser = ProfilingParser(logs)
    profiling_data = parser.process()

    if not proc.is_ok:
        raise ClientErrorException("Profiling failed during execution.")

    execution_details = {f"{model}_profiling": profiling_data}
    workdir.update_execution_details(
        request_id=request_id,
        execution_details=execution_details,
    )
    response_data["execution_details"].update(
        {f"{model}_profiling": profiling_data},
    )
    return response_data
