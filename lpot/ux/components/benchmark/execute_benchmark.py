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
import os
from typing import Any, Dict, List

from lpot.ux.components.benchmark import Benchmarks
from lpot.ux.components.benchmark.benchmark import Benchmark
from lpot.ux.utils.exceptions import ClientErrorException, InternalException
from lpot.ux.utils.executor import Executor
from lpot.ux.utils.logger import log
from lpot.ux.utils.parser import BenchmarkParserFactory
from lpot.ux.utils.templates.workdir import Workdir
from lpot.ux.utils.utils import _load_json_as_dict
from lpot.ux.utils.workload.workload import Workload
from lpot.ux.web.communication import MessageQueue

mq = MessageQueue()


def execute_benchmark(data: Dict[str, Any]) -> None:
    """
    Execute benchmark.

    Expected data:
    {
        "id": "configuration_id",
        "workspace_path": "/path/to/workspace",
        "input_model": {
            "precision": "fp32",
            "path": "/localdisk/fp32.pb"
        },
        "optimized_model": {
            "precision": "int8",
            "path": "/localdisk/int8.pb"
        }
    }
    """
    from lpot.ux.utils.workload.workload import Workload

    request_id = str(data.get("id", ""))
    input_model = data.get("input_model", None)
    input_model.update({"model_type": "input_model"})

    optimized_model = data.get("optimized_model", None)
    optimized_model.update({"model_type": "optimized_model"})

    if not (request_id and input_model and optimized_model):
        message = "Missing request id, input or optimized model data."
        mq.post_error(
            "benchmark_finish",
            {"message": message, "code": 404, "id": request_id},
        )
        raise ClientErrorException(message)

    workdir = Workdir(request_id=request_id, overwrite=False)
    try:
        workload_path = workdir.workload_path
        workload_data = _load_json_as_dict(
            os.path.join(workload_path, "workload.json"),
        )
    except Exception as err:
        mq.post_error(
            "benchmark_finish",
            {"message": repr(err), "code": 404, "id": request_id},
        )
        raise ClientErrorException(repr(err))

    workload = Workload(workload_data)

    response_data: Dict[str, Any] = {"id": request_id, "execution_details": {}}

    mq.post_success(
        "benchmark_start",
        {
            "message": "started",
            "id": request_id,
        },
    )

    models = [input_model, optimized_model]

    benchmark_count = 0
    benchmark_total = 0

    for model_info in models:
        benchmark_modes: List[str] = model_info.get("mode", [Benchmarks.PERF])
        if (
            not workload.tune and Benchmarks.ACC not in benchmark_modes
        ):  # Accuracy information is provided only in tuning
            benchmark_modes.append(Benchmarks.ACC)
        model_info.update({"benchmark_modes": benchmark_modes})
        benchmark_total += len(benchmark_modes)

    for model_info in models:
        model_precision = model_info.get("precision", None)
        model_type = model_info.get("model_type", None)
        model_path = model_info.get("path", None)
        benchmark_modes = model_info.get("benchmark_modes", None)

        if not (model_precision and model_path and model_type and benchmark_modes):
            message = "Missing model precision, model path or model type."
            mq.post_error(
                "benchmark_finish",
                {"message": message, "code": 404, "id": request_id},
            )
            raise ClientErrorException(message)

        for benchmark_mode in benchmark_modes:
            benchmark_count += 1
            response_data = benchmark_model_and_respond_to_ui(
                response_data=response_data,
                workload=workload,
                workdir=workdir,
                model=model_type,
                model_path=model_path,
                model_precision=model_precision,
                benchmark_mode=benchmark_mode,
                benchmark_count=benchmark_count,
                benchmark_total=benchmark_total,
            )


def benchmark_model_and_respond_to_ui(
    response_data: dict,
    workload: Workload,
    workdir: Workdir,
    model: str,
    model_path: str,
    model_precision: str,
    benchmark_mode: str,
    benchmark_count: int,
    benchmark_total: int,
) -> dict:
    """Benchmark model and update UI."""
    try:
        response_data = benchmark_model(
            response_data=response_data,
            workload=workload,
            workdir=workdir,
            model=model,
            model_path=model_path,
            model_precision=model_precision,
            benchmark_mode=benchmark_mode,
            benchmark_count=benchmark_count,
            benchmark_total=benchmark_total,
        )
        mq.post_success("benchmark_finish", response_data)
        return response_data
    except ClientErrorException as err:
        log.error(f"Benchmark failed: {str(err)}.")
        mq.post_failure("benchmark_finish", {"message": "failed", "id": workload.id})
        raise


def benchmark_model(
    response_data: dict,
    workload: Workload,
    workdir: Workdir,
    model: str,
    model_path: str,
    model_precision: str,
    benchmark_mode: str,
    benchmark_count: int,
    benchmark_total: int,
) -> dict:
    """Benchmark model and prepare response data."""
    request_id = response_data.get("id")

    benchmark: Benchmark = Benchmark(
        workload=workload,
        model_path=model_path,
        precision=model_precision,
        mode=benchmark_mode,
    )

    log_name = f"{model}_{benchmark_mode}_benchmark"

    executor = Executor(
        workload.workload_path,
        subject="benchmark",
        data={"id": request_id},
        send_response=False,
        log_name=log_name,
        additional_log_names=["output.txt"],
    )

    proc = executor.call(
        benchmark.command,
    )

    logs = [os.path.join(workload.workload_path, f"{log_name}.txt")]

    if not proc.is_ok:
        raise ClientErrorException("Benchmark failed during execution.")

    parser = BenchmarkParserFactory.get_parser(benchmark_mode, logs)
    metrics = parser.process()
    metric = {}
    execution_details: Dict[str, Any] = {}

    if benchmark_mode == Benchmarks.PERF:
        result_field = f"perf_throughput_{model}"
    elif benchmark_mode == Benchmarks.ACC:
        result_field = f"acc_{model}"
    else:
        raise InternalException(f"Benchmark mode {benchmark_mode} is not supported.")

    if isinstance(metrics, dict):
        metric = {result_field: metrics.get(result_field, "")}
        execution_details = response_data.get("execution_details", {})
        model_benchmark_details = execution_details.get(f"{model}_benchmark", {})
        model_benchmark_details.update(
            {
                benchmark_mode: benchmark.serialize(),
            },
        )

        response_data.update({"progress": f"{benchmark_count}/{benchmark_total}"})
        response_data.update(metric)
        response_data["execution_details"].update(
            {f"{model}_benchmark": model_benchmark_details},
        )
    workdir.update_metrics(
        request_id=request_id,
        metric_data=metric,
    )
    workdir.update_execution_details(
        request_id=request_id,
        execution_details=execution_details,
    )
    log.debug(f"Parsed data is {json.dumps(response_data)}")
    mq.post_success("benchmark_progress", response_data)

    return response_data
