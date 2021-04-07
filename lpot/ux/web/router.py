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

"""Connector between api.py and components."""

import os
from threading import Thread
from typing import Any, Dict

from lpot.ux.components.benchmark.execute_benchmark import execute_benchmark
from lpot.ux.components.configuration_wizard.get_boundary_nodes import get_boundary_nodes
from lpot.ux.components.configuration_wizard.get_configuration import get_predefined_configuration
from lpot.ux.components.configuration_wizard.params_feeder import (
    get_possible_values,
    get_possible_values_v2,
)
from lpot.ux.components.configuration_wizard.save_workload import save_workload
from lpot.ux.components.file_browser.file_browser import get_directory_entries
from lpot.ux.components.graph.graph import Graph
from lpot.ux.components.graph.graph_reader import GraphReader
from lpot.ux.components.manage_workspace import (
    delete_workload,
    get_default_path,
    get_workloads_list,
    set_workspace,
)
from lpot.ux.components.model_zoo.download_config import download_config
from lpot.ux.components.model_zoo.download_model import download_model
from lpot.ux.components.model_zoo.list_models import list_models
from lpot.ux.components.tune.execute_tune import execute_tuning
from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.hw_info import HWInfo
from lpot.ux.utils.json_serializer import JsonSerializer
from lpot.ux.utils.templates.workdir import Workdir
from lpot.ux.web.communication import Request, Response, create_simple_response
from lpot.ux.web.exceptions import ServiceNotFoundException


class Router:
    """Connector between api.py and components."""

    def __init__(self) -> None:
        """Initialize object."""
        clean_workloads_wip_status()

    def handle(self, request: Request) -> Response:
        """Run operation on requested component and return result."""
        operation_map = {
            "filesystem": get_directory_entries,
            "save_workload": save_workload,
            "delete_workload": delete_workload,
            "configuration": get_predefined_configuration,
            "tune": process_request_for_tuning,
            "benchmark": process_request_for_benchmark,
            "get_default_path": get_default_path,
            "set_workspace": set_workspace,
            "get_workloads_list": get_workloads_list,
            "get_boundary_nodes": process_request_for_boundary_nodes,
            "get_possible_values": get_possible_values,
            "get_possible_values_v2": get_possible_values_v2,
            "download_model": process_request_for_model_download,
            "list_model_zoo": list_models,
            "download_config": process_request_for_model_config,
            "model_graph_readable": can_read_model_graph,
            "model_graph": get_model_graph,
            "system_info": get_system_info,
        }

        operation = operation_map.get(request.operation)
        if operation is None:
            raise ServiceNotFoundException(f"Unable to find {request.operation}")

        data = operation(request.data)

        serialized_data = JsonSerializer.serialize_item(data)

        return create_simple_response(serialized_data)


def process_request_for_tuning(data: dict) -> dict:
    """Set thread and execute tuning."""
    t = Thread(target=_execute_tuning_benchmark, args=(data,))
    t.daemon = True
    t.start()

    return {"exit_code": 102, "message": "processing"}


def process_request_for_benchmark(data: dict) -> dict:
    """Set thread and execute tuning."""
    t = Thread(target=execute_benchmark, args=(data,))
    t.daemon = True
    t.start()

    return {"exit_code": 102, "message": "processing"}


def process_request_for_boundary_nodes(data: dict) -> dict:
    """Set thread and get boundary nodes."""
    t = Thread(target=get_boundary_nodes, args=(data,))
    t.daemon = True
    t.start()

    return {"exit_code": 102, "message": "processing"}


def process_request_for_model_download(data: dict) -> dict:
    """Set thread and download model."""
    t = Thread(target=download_model, args=(data,))
    t.daemon = True
    t.start()

    return {"exit_code": 102, "message": "processing"}


def process_request_for_model_config(data: dict) -> dict:
    """Set thread and download model."""
    t = Thread(target=download_config, args=(data,))
    t.daemon = True
    t.start()

    return {"exit_code": 102, "message": "processing"}


def _execute_tuning_benchmark(data: dict) -> None:
    """Execute both tuning and benchmark."""
    tuning_data = execute_tuning(data)
    benchmark_data = {
        "id": data.get("id"),
        "models": [
            {
                "precision": "fp32",
                "path": tuning_data.get("execution_details", {})
                .get("tuning", {})
                .get("model_path"),
            },
            {
                "precision": "int8",
                "path": tuning_data.get("execution_details", {})
                .get("tuning", {})
                .get(
                    "model_output_path",
                ),
            },
        ],
        "workspace_path": data.get("workspace_path"),
    }
    if not tuning_data.get("is_custom_dataloader", None):
        execute_benchmark(benchmark_data)


def clean_workloads_wip_status() -> None:
    """Clean WIP status for workloads in workloads_list.json."""
    workdir = Workdir(workspace_path=os.environ["HOME"])
    workdir.clean_status(status_to_clean="wip")


def can_read_model_graph(data: Dict[str, Any]) -> str:
    """Check if can read model graph."""
    graph_reader = GraphReader()
    graph_reader.ensure_model_readable(_get_string_value(data, "path"))
    return "OK"


def get_model_graph(data: Dict[str, Any]) -> Graph:
    """Get model graph."""
    graph_reader = GraphReader()
    return graph_reader.read(_get_string_value(data, "path"))


def _get_string_value(data: Dict[str, Any], name: str) -> str:
    """Get string value from request."""
    try:
        return data[name][0]
    except KeyError:
        raise ClientErrorException(f"Missing {name} parameter")


def get_system_info(data: Dict[str, Any]) -> dict:
    """Get system info."""
    hw_info = vars(HWInfo())
    if "cores" in hw_info:
        del hw_info["cores"]
    return hw_info
