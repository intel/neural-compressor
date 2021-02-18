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

"""Execute tune."""

import json
import os
from typing import Any, Dict

from lpot.ux.components.tune.tuning import Tuning
from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.executor import Executor
from lpot.ux.utils.logger import log
from lpot.ux.utils.parser import Parser
from lpot.ux.utils.templates.workdir import Workdir
from lpot.ux.utils.utils import get_size, load_json
from lpot.ux.web.communication import MessageQueue

mq = MessageQueue()


def execute_tuning(data: Dict[str, Any]) -> dict:
    """Get configuration."""
    from lpot.ux.utils.workload.workload import Workload

    if not str(data.get("id", "")):
        message = "Missing request id."
        mq.post_error(
            "tuning_finish",
            {"message": message, "code": 404},
        )
        raise Exception(message)

    request_id: str = data["id"]
    workdir = Workdir(request_id=request_id)
    workload_path: str = workdir.workload_path
    try:
        workload_data = load_json(
            os.path.join(workload_path, "workload.json"),
        )
    except Exception as err:
        mq.post_error(
            "tuning_finish",
            {"message": repr(err), "code": 404, "id": request_id},
        )
        raise err
    workload = Workload(workload_data)
    tuning: Tuning = Tuning(workload, workdir.workload_path, workdir.template_path)
    send_data = {
        "message": "started",
        "id": request_id,
        "size_fp32": get_size(tuning.model_path),
    }
    workdir.clean_logs()
    workdir.update_data(
        request_id=request_id,
        model_path=tuning.model_path,
        model_output_path=tuning.model_output_path,
        status="wip",
    )

    executor = Executor(
        workspace_path=workload_path,
        subject="tuning",
        data=send_data,
        log_name="output",
    )

    proc = executor.call(
        tuning.command,
    )
    tuning_time = executor.process_duration
    if tuning_time:
        tuning_time = round(tuning_time, 2)
    log.debug(f"Elapsed time: {tuning_time}")
    logs = [os.path.join(workload_path, "output.txt")]
    parser = Parser(logs)
    if proc.is_ok:
        response_data = parser.process()

        if isinstance(response_data, dict):
            response_data["id"] = request_id
            response_data["tuning_time"] = tuning_time
            response_data["size_int8"] = get_size(tuning.model_output_path)
            response_data["model_output_path"] = tuning.model_output_path
            response_data["size_fp32"] = get_size(tuning.model_path)
            response_data["is_custom_dataloader"] = bool(workdir.template_path)

            workdir.update_data(
                request_id=request_id,
                model_path=tuning.model_path,
                model_output_path=tuning.model_output_path,
                metric=response_data,
                status="success",
                execution_details={"tuning": tuning.serialize()},
            )
            response_data["execution_details"] = {"tuning": tuning.serialize()}

        log.debug(f"Parsed data is {json.dumps(response_data)}")
        mq.post_success("tuning_finish", response_data)
        return response_data
    else:
        log.debug("FAIL")
        workdir.update_data(
            request_id=request_id,
            model_path=tuning.model_path,
            status="error",
        )
        mq.post_failure("tuning_finish", {"message": "failed", "id": request_id})
        raise ClientErrorException("Tuning failed during execution.")
