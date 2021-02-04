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
import logging as log
import os
from typing import Any, Dict

from lpot.ux.components.lpot_tune.tuning import Tuning
from lpot.ux.utils.executor import Executor
from lpot.ux.utils.parser import Parser
from lpot.ux.utils.utils import get_size, load_json
from lpot.ux.web.communication import MessageQueue

log.basicConfig(level=log.INFO)
mq = MessageQueue()


def execute_tuning(data: Dict[str, Any]) -> None:
    """Get configuration."""
    from lpot.ux.utils.workload.workload import Workload

    request_id = data.get("id", None)
    workspace_path = data.get("workspace_path", None)

    if not (request_id and workspace_path):
        message = "Missing workspace path or request id."
        mq.post_error(
            "tuning_finish",
            {"message": message, "code": 404, "id": request_id},
        )
        raise Exception(message)

    try:
        workload_data = load_json(
            os.path.join(workspace_path, f"workload.{request_id}.json"),
        )
    except Exception as err:
        mq.post_error(
            "tuning_finish",
            {"message": repr(err), "code": 404, "id": request_id},
        )
        raise err
    workload = Workload(workload_data)
    tuning: Tuning = Tuning(workload)
    send_data = {
        "message": "started",
        "id": request_id,
        "size_fp32": get_size(tuning.model_path),
    }
    executor = Executor(workspace_path, subject="tuning", data=send_data)

    proc = executor.call(
        tuning.command,
    )
    tuning_time = executor.process_duration
    if tuning_time:
        tuning_time = round(tuning_time, 2)
    log.info(f"Elapsed time: {tuning_time}")
    logs = [os.path.join(workspace_path, f"{request_id}.txt")]
    parser = Parser(logs)
    if proc.is_ok:
        response_data = parser.process()
        if isinstance(response_data, dict):
            response_data["id"] = request_id
            response_data["tuning_time"] = f"{tuning_time}s"
            response_data["size_int8"] = get_size(tuning.model_output_path)
            response_data["model_output_path"] = tuning.model_output_path

        log.info(f"Parsed data is {json.dumps(response_data)}")
        mq.post_success("tuning_finish", response_data)
    else:
        log.info("FAIL")
        mq.post_failure("tuning_finish", {"message": "failed", "id": request_id})
