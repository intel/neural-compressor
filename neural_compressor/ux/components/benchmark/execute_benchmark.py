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
import shutil
import subprocess
import time
from typing import Any, Dict

from neural_coder import bench, enable
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


def benchmark_result_update(
    benchmark: Any,
    logs: Any,
    executor: Any,
    request_id: Any,
) -> dict:
    """Update benchmark result. (TF, ONNX models)."""
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


def benchmark_result_update_pytorch_script(
    benchmark: Any,
    request_id: Any,
    neural_coder_performance: Any,
    neural_coder_time: float,
) -> dict:
    """Update benchmark result. (PyTorch scripted models)."""
    BenchmarkAPIInterface.add_result(
        {
            "benchmark_id": benchmark.benchmark_id,
            "accuracy": "",
            "performance": neural_coder_performance[0],
        },
    )

    BenchmarkAPIInterface.update_benchmark_duration(
        {
            "id": benchmark.benchmark_id,
            "duration": neural_coder_time,
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


def pytorch_script_bench_optimizations(
    benchmark: Any,
    command_line_of_script: str,
    logs: Any,
) -> float:
    """Benchmark for PyTorch scripted models (on the optimizations)."""
    copy_model_path = benchmark.model_path[:-23] + "copy_model.py"
    copy_model_backup_path = benchmark.model_path[:-23] + "copy_model_backup.py"
    if os.path.exists(copy_model_backup_path):
        os.remove(copy_model_path)
        os.rename(copy_model_backup_path, copy_model_path)
    shutil.copy(copy_model_path, copy_model_backup_path)
    optimization_patch_path = benchmark.model_path
    # apply optimization patch
    # TODO: Consider using some python library
    # instead of calling patch directly to make it OS independent.
    sp_patch = subprocess.Popen(
        "patch -d/ -p0 < " + optimization_patch_path,
        env=os.environ,
        shell=True,  # nosec
        stdout=subprocess.PIPE,
    )
    sp_patch.wait()
    # enable batch size and benchmark
    enable(
        code=copy_model_path,
        features=["pytorch_benchmark", "pytorch_change_batch_size"],
        target_batch_size=benchmark.batch_size,
        overwrite=True,
        logging_level="critical",
    )
    # benchmark
    # TODO: enabling specification of log dir path in Neural Coder in future releases
    neural_coder_performance, mode, ws_path = bench(
        code=copy_model_path,
        entry_code_args=command_line_of_script,
        mode="self_defined",
        ninstances=benchmark.num_of_instance,
        ncore_per_instance=benchmark.cores_per_instance,
        bench_batch_size=benchmark.batch_size,
        logging_level="critical",
    )
    shutil.copy(ws_path + "/bench.log", logs[0])

    return neural_coder_performance


def pytorch_script_bench_original_model(
    benchmark: Any,
    command_line_of_script: str,
    logs: Any,
) -> float:
    """Benchmark for PyTorch scripted models (on the original model)."""
    copy_original_model_path = benchmark.original_model_path[:-3] + "_copy.py"
    shutil.copy(benchmark.original_model_path, copy_original_model_path)
    # enable batch size and benchmark
    enable(
        code=copy_original_model_path,
        features=["pytorch_benchmark", "pytorch_change_batch_size"],
        target_batch_size=benchmark.batch_size,
        overwrite=True,
        logging_level="critical",
    )
    # benchmark
    # TODO: enabling specification of log dir path in Neural Coder in future releases
    neural_coder_performance, mode, ws_path = bench(
        code=copy_original_model_path,
        entry_code_args=command_line_of_script,
        mode="self_defined",
        ninstances=benchmark.num_of_instance,
        ncore_per_instance=benchmark.cores_per_instance,
        bench_batch_size=benchmark.batch_size,
        logging_level="critical",
    )
    shutil.copy(ws_path + "/bench.log", logs[0])
    os.remove(copy_original_model_path)

    return neural_coder_performance


def execute_real_benchmark(
    request_id: str,
    project_details: dict,
    benchmark_details: dict,
) -> dict:
    """Execute benchmark."""
    benchmark: Benchmark = Benchmark(project_details, benchmark_details)
    command_line_of_script = benchmark.command
    is_pytorch_script = False
    if benchmark.framework == "pytorch":
        is_pytorch_script = True
        benchmark.command = "from neural_coder import bench"
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

    if is_pytorch_script:
        time_start = time.time()
        if benchmark.model_path[-5:] == ".diff":
            neural_coder_performance = pytorch_script_bench_optimizations(
                benchmark,
                command_line_of_script,
                logs,
            )
        else:
            neural_coder_performance = pytorch_script_bench_original_model(
                benchmark,
                command_line_of_script,
                logs,
            )

        time_end = time.time()
        neural_coder_time = time_end - time_start
    else:
        proc = executor.call(
            benchmark.command,
        )

    if not is_pytorch_script:
        if not proc.is_ok:
            raise InternalException("Benchmark failed during execution.")

    if is_pytorch_script:
        response_data = benchmark_result_update_pytorch_script(
            benchmark,
            request_id,
            neural_coder_performance,
            neural_coder_time,
        )
    else:
        response_data = benchmark_result_update(
            benchmark,
            logs,
            executor,
            request_id,
        )

    return response_data
