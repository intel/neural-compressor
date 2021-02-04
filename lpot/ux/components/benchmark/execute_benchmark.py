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

"""Execute benchmark."""

import json
import logging as log
import os
from typing import Any, Dict

from lpot.ux.components.benchmark.benchmark import Benchmark
from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.executor import Executor
from lpot.ux.utils.parser import Parser
from lpot.ux.utils.utils import load_json
from lpot.ux.web.communication import MessageQueue

log.basicConfig(level=log.INFO)
mq = MessageQueue()


def execute_benchmark(data: Dict[str, Any]) -> None:
    """
    Execute benchmark.

    Expected data:
    {
        "id": "configuration_id",
        "workspace_path": "/path/to/workspace",
        "models": [
            {
                "precision": "fp32",
                "path": "/localdisk/fp32.pb"
            },
            {
                "precision": "int8",
                "path": "/localdisk/int8.pb"
            }
        ]
    }
    """
    from lpot.ux.utils.workload.workload import Workload

    request_id = data.get("id", None)
    workspace_path = data.get("workspace_path", None)
    models = data.get("models", None)

    if not (request_id and workspace_path and models):
        message = "Missing workspace path, request id or model list."
        mq.post_error(
            "benchmark_finish",
            {"message": message, "code": 404, "id": request_id},
        )
        raise ClientErrorException(message)

    try:
        workload_data = load_json(
            os.path.join(workspace_path, f"workload.{request_id}.json"),
        )
    except Exception as err:
        mq.post_error(
            "benchmark_finish",
            {"message": repr(err), "code": 404, "id": request_id},
        )
        raise ClientErrorException(repr(err))

    workload = Workload(workload_data)

    response_data = {
        "id": request_id,
    }

    mq.post_success(
        "benchmark_start",
        {
            "message": "started",
            "id": request_id,
        },
    )

    for idx, model_info in enumerate(models, start=1):
        model_precision = model_info.get("precision", None)
        model_path = model_info.get("path", None)
        benchmark_mode = model_info.get("mode", "performance")
        if not (model_precision and model_path):
            message = "Missing model precision or model path."
            mq.post_error(
                "benchmark_finish",
                {"message": message, "code": 404, "id": request_id},
            )
            raise ClientErrorException(message)

        benchmark: Benchmark = Benchmark(
            workload=workload,
            model_path=model_path,
            datatype=model_precision,
            mode=benchmark_mode,
        )

        log_name = f"{request_id}_{model_precision}_{benchmark_mode}_benchmark"

        executor = Executor(
            workspace_path,
            subject="benchmark",
            data={"id": request_id},
            send_response=False,
            log_name=log_name,
        )

        proc = executor.call(
            benchmark.command,
        )

        logs = [os.path.join(workspace_path, f"{log_name}.txt")]

        if proc.is_ok:
            parser = Parser(logs)
            metrics = parser.process()
            throughput_field = f"perf_throughput_{model_precision}"
            if isinstance(metrics, dict):
                response_data.update(
                    {
                        throughput_field: metrics.get(throughput_field),
                        "progress": f"{idx}/{len(models)}",
                    },
                )
            log.info(f"Parsed data is {json.dumps(response_data)}")
            mq.post_success("benchmark_progress", response_data)
        else:
            log.error("Benchmark failed.")
            mq.post_failure("benchmark_finish", {"message": "failed", "id": request_id})
            return

    mq.post_success("benchmark_finish", response_data)
