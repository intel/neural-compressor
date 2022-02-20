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

import os
from typing import Any, Dict

from neural_compressor.ux.components.benchmark.benchmark import Benchmark
from neural_compressor.ux.components.db_manager.db_operations import (
    BenchmarkAPIInterface,
    ProjectAPIInterface,
)
from neural_compressor.ux.utils.consts import ExecutionStatus
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.executor import Executor
from neural_compressor.ux.utils.parser import BenchmarkParserFactory
from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.web.communication import MessageQueue

mq = MessageQueue()


def execute_benchmark(data: Dict[str, Any]) -> None:
    """
    Execute benchmark.

    Expected data:
    {
        "request_id": "asd",
        "benchmark_id": "1"
    }
    """
    if not all([str(data.get("request_id", "")), str(data.get("benchmark_id", ""))]):
        message = "Missing request id or benchmark id."
        mq.post_error(
            "benchmark_finish",
            {"message": "Failure", "code": 404},
        )
        raise ClientErrorException(message)

    request_id: str = str(data["request_id"])
    benchmark_id: int = int(data["benchmark_id"])

    try:
        benchmark_details = BenchmarkAPIInterface.get_benchmark_details({"id": benchmark_id})
        # Temporary workaround for benchmarks mode
        project_id = benchmark_details["project_id"]
        project_details = ProjectAPIInterface.get_project_details({"id": project_id})

        BenchmarkAPIInterface.update_benchmark_status(
            {
                "id": benchmark_id,
                "status": ExecutionStatus.WIP,
            },
        )

        response_data = execute_real_benchmark(
            request_id=request_id,
            project_details=project_details,
            benchmark_details=benchmark_details,
        )

        mq.post_success("benchmark_finish", response_data)
    except Exception:
        BenchmarkAPIInterface.update_benchmark_status(
            {
                "id": benchmark_id,
                "status": ExecutionStatus.ERROR,
            },
        )
        mq.post_error(
            "benchmark_finish",
            {"message": "Failure", "code": 404, "request_id": request_id},
        )
        raise


def execute_real_benchmark(
    request_id: str,
    project_details: dict,
    benchmark_details: dict,
) -> dict:
    """Execute benchmark."""
    benchmark: Benchmark = Benchmark(project_details, benchmark_details)

    Workdir.clean_logs(benchmark.workdir)
    benchmark.generate_config()

    logs = [os.path.join(benchmark.workdir, "output.txt")]

    BenchmarkAPIInterface.update_paths(
        {
            "id": benchmark.benchmark_id,
            "config_path": benchmark.config_path,
            "log_path": logs[0],
        },
    )

    BenchmarkAPIInterface.update_execution_command(
        {
            "id": benchmark.benchmark_id,
            "execution_command": benchmark.command,
        },
    )

    send_data = {
        "message": "started",
        "request_id": request_id,
        "config_path": benchmark.config_path,
        "output_path": logs[0],
    }

    executor = Executor(
        workspace_path=benchmark.workdir,
        subject="benchmark",
        data=send_data,
        log_name="output",
    )

    proc = executor.call(
        benchmark.command,
    )

    if not proc.is_ok:
        raise InternalException("Benchmark failed during execution.")

    parser = BenchmarkParserFactory.get_parser(benchmark.mode, logs)
    metrics = parser.process()

    BenchmarkAPIInterface.add_result(
        {
            "benchmark_id": benchmark.benchmark_id,
            "accuracy": metrics.get("accuracy"),
            "performance": metrics.get("throughput"),
        },
    )

    BenchmarkAPIInterface.update_benchmark_duration(
        {
            "id": benchmark.benchmark_id,
            "duration": executor.process_duration,
        },
    )

    BenchmarkAPIInterface.update_benchmark_status(
        {
            "id": benchmark.benchmark_id,
            "status": ExecutionStatus.SUCCESS,
        },
    )

    benchmark_data = BenchmarkAPIInterface.get_benchmark_details(
        {
            "id": benchmark.benchmark_id,
        },
    )
    response_data = {
        "request_id": request_id,
    }
    response_data.update(benchmark_data)

    return response_data
