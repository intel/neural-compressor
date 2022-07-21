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

"""Execute tune."""

import os
import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional

from neural_compressor.ux.components.db_manager.db_operations import (
    DatasetAPIInterface,
    ModelAPIInterface,
    OptimizationAPIInterface,
    ProjectAPIInterface,
)
from neural_compressor.ux.components.names_mapper.names_mapper import MappingDirection, NamesMapper
from neural_compressor.ux.components.optimization.factory import OptimizationFactory
from neural_compressor.ux.components.optimization.neural_coder_optimization.optimize_model import (
    optimize_pt_script,
)
from neural_compressor.ux.components.optimization.optimization import Optimization
from neural_compressor.ux.components.optimization.tune.tuning import Tuning
from neural_compressor.ux.components.optimization.tuning_history import Watcher
from neural_compressor.ux.utils.consts import ExecutionStatus
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.executor import Executor
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.parser import OptimizationParser
from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.utils.utils import get_size, normalize_string
from neural_compressor.ux.web.communication import MessageQueue

mq = MessageQueue()


def execute_optimization_pytorch_script(
    optimization: Any,
    request_id: Any,
    optimization_id: Any,
) -> Any:
    """Optimize PyTorch scripted models."""
    logs = [os.path.join(optimization.workdir, "neural_coder_patch.diff")]

    Workdir.clean_logs(optimization.workdir)
    OptimizationAPIInterface.update_optimization_status(
        {
            "id": optimization_id,
            "status": ExecutionStatus.WIP,
        },
    )

    optimization.generate_config()

    OptimizationAPIInterface.update_paths(
        {
            "id": optimization_id,
            "config_path": optimization.config_path,
            "log_path": logs[0],
        },
    )

    if optimization.output_precision == "int8 static quantization":
        optimization.command = [
            "from neural_coder import enable; enable(code="
            + optimization.workdir
            + "/copy_model.py"
            + ", features=['pytorch_inc_static_quant'], "
            + "overwrite=False, save_patch_path="
            + optimization.workdir
            + "/"
            + ")",
        ]
    else:
        optimization.command = [
            "from neural_coder import enable; enable(code="
            + optimization.workdir
            + "/copy_model.py"
            + ", features=['pytorch_inc_dynamic_quant'], "
            + "overwrite=False, save_patch_path="
            + optimization.workdir
            + "/"
            + ")",
        ]

    OptimizationAPIInterface.update_execution_command(
        {
            "id": optimization_id,
            "execution_command": optimization.command,
        },
    )

    optimization_time = optimize_pt_script(optimization)

    logs = [os.path.join(optimization.workdir, "neural_coder_patch.diff")]

    return optimization, optimization_time, logs


def execute_optimization_regular(
    optimization: Any,
    request_id: Any,
    optimization_id: Any,
) -> Any:
    """Optimize PB/ONNX models."""
    logs = [os.path.join(optimization.workdir, "output.txt")]
    send_data = {
        "message": "started",
        "request_id": request_id,
        "size_input_model": get_size(optimization.input_graph),
        "config_path": optimization.config_path,
        "output_path": logs[0],
    }

    Workdir.clean_logs(optimization.workdir)
    OptimizationAPIInterface.update_optimization_status(
        {
            "id": optimization_id,
            "status": ExecutionStatus.WIP,
        },
    )

    optimization.generate_config()

    OptimizationAPIInterface.update_paths(
        {
            "id": optimization_id,
            "config_path": optimization.config_path,
            "log_path": logs[0],
        },
    )

    OptimizationAPIInterface.update_execution_command(
        {
            "id": optimization_id,
            "execution_command": optimization.command,
        },
    )

    executor = Executor(
        workspace_path=optimization.workdir,
        subject="optimization",
        data=send_data,
        log_name="output",
    )

    collect_tuning_history: bool = check_if_collect_tuning_history(optimization)

    if collect_tuning_history:
        tuning_history_watcher = Watcher(request_id, optimization)
        threading.Thread(target=tuning_history_watcher, daemon=True).start()

    proc = executor.call(optimization.command)

    if collect_tuning_history:
        tuning_history_watcher.stop(process_succeeded=proc.is_ok)

    optimization_time = executor.process_duration
    if optimization_time:
        optimization_time = round(optimization_time, 2)
    log.debug(f"Elapsed time: {optimization_time}")

    logs = [os.path.join(optimization.workdir, "output.txt")]

    return optimization, optimization_time, logs, proc


def execute_optimization(data: Dict[str, Any]) -> dict:
    """Get configuration."""
    if not all([str(data.get("request_id", "")), str(data.get("optimization_id", ""))]):
        message = "Missing request id or optimization id."
        mq.post_error(
            "optimization_finish",
            {"message": message, "code": 404},
        )
        raise Exception(message)

    request_id: str = str(data["request_id"])
    optimization_id: int = int(data["optimization_id"])

    project_id: Optional[int] = None
    try:
        optimization_details = OptimizationAPIInterface.get_optimization_details(
            {"id": optimization_id},
        )
        dataset_details = DatasetAPIInterface.get_dataset_details(
            {"id": optimization_details["dataset"]["id"]},
        )
        project_id = optimization_details["project_id"]
        project_details = ProjectAPIInterface.get_project_details({"id": project_id})
        optimization: Optimization = OptimizationFactory.get_optimization(
            optimization_data=optimization_details,
            project_data=project_details,
            dataset_data=dataset_details,
        )

        # execute optimization
        is_pytorch_script = False
        if optimization.framework == "pytorch":
            is_pytorch_script = True
            optimization, optimization_time, logs = execute_optimization_pytorch_script(
                optimization,
                request_id,
                optimization_id,
            )
        else:
            optimization, optimization_time, logs, proc = execute_optimization_regular(
                optimization,
                request_id,
                optimization_id,
            )

        # update optimization result
        if is_pytorch_script or proc.is_ok:
            optimized_model_data = parse_logs(
                optimization=optimization,
                optimization_details=optimization_details,
                project_details=project_details,
                logs=logs,
            )
            if is_pytorch_script:
                optimized_model_data["model_path"] = logs[0]

            optimized_model_id = ModelAPIInterface.add_model(optimized_model_data)
            OptimizationAPIInterface.update_optimized_model(
                {
                    "id": optimization_id,
                    "optimized_model_id": optimized_model_id,
                },
            )
            OptimizationAPIInterface.update_optimization_duration(
                {
                    "id": optimization_id,
                    "duration": optimization_time,
                },
            )
            OptimizationAPIInterface.update_optimization_status(
                {
                    "id": optimization_id,
                    "status": ExecutionStatus.SUCCESS,
                },
            )

            optimization_data = OptimizationAPIInterface.get_optimization_details(
                {
                    "id": optimization_id,
                },
            )
            response_data = {
                "request_id": request_id,
            }
            response_data.update(optimization_data)

            mq.post_success("optimization_finish", response_data)
            return response_data

        log.debug("FAIL")
        OptimizationAPIInterface.update_optimization_status(
            {
                "id": optimization_id,
                "status": ExecutionStatus.ERROR,
            },
        )
        raise ClientErrorException("Optimization failed during execution.")
    except Exception as err:
        mq.post_failure(
            "optimization_finish",
            {
                "message": str(err),
                "request_id": request_id,
                "id": optimization_id,
                "project_id": project_id,
            },
        )
        optimization_status = OptimizationAPIInterface.get_optimization_details(
            {"id": optimization_id},
        ).get("status")
        if optimization_status != ExecutionStatus.ERROR.value:
            OptimizationAPIInterface.update_optimization_status(
                {
                    "id": optimization_id,
                    "status": ExecutionStatus.ERROR,
                },
            )
        raise


def parse_model_data_to_bench_names(data: dict) -> dict:
    """Parse names to Bench format."""
    model_data = deepcopy(data)
    names_mapper = NamesMapper(MappingDirection.ToBench)

    framework = model_data.get("framework", None)
    if framework is not None:
        mapped_framework = names_mapper.map_name(
            parameter_type="framework",
            value=framework,
        )
        model_data.update({"framework": mapped_framework})
    return model_data


def parse_logs(
    optimization: Optimization,
    optimization_details: dict,
    project_details: dict,
    logs: List[str],
) -> dict:
    """Parse optimization logs."""
    parser = OptimizationParser(logs)
    parsed_log = parser.process()

    optimized_model_path = parsed_log.get(
        "path_optimized_model",
        optimization.output_graph,
    )
    optimization.output_graph = optimized_model_path

    if not isinstance(parsed_log, dict):
        raise InternalException("Failure with parsing logs.")
    optimized_model_data: dict = {
        "project_id": optimization.project_id,
        "model_name": normalize_string(optimization_details["name"]),
        "model_path": optimized_model_path,
        "framework": optimization.framework,
        "size": get_size(optimized_model_path),
        "precision_id": optimization_details["precision"]["id"],
        "domain_id": project_details["input_model"]["domain"]["id"],
        "domain_flavour_id": project_details["input_model"]["domain_flavour"]["id"],
        "input_nodes": optimization.input_nodes,
        "output_nodes": optimization.output_nodes,
        "supports_profiling": project_details["input_model"]["supports_profiling"],
        "supports_graph": project_details["input_model"]["supports_graph"],
    }

    optimized_model_data = parse_model_data_to_bench_names(optimized_model_data)
    return optimized_model_data


def check_if_collect_tuning_history(optimization: Optimization) -> bool:
    """Check whether tuning history can be collected for specified optimization."""
    collect_tuning_history: bool = False
    if isinstance(optimization, Tuning):
        collect_tuning_history = True
    return collect_tuning_history
