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
from threading import Thread
from typing import Any, Callable, Dict

from werkzeug.wrappers import Response as WebResponse

from neural_compressor.ux.components.benchmark.execute_benchmark import execute_benchmark
from neural_compressor.ux.components.configuration_wizard.get_boundary_nodes import get_boundary_nodes
from neural_compressor.ux.components.configuration_wizard.get_configuration import get_predefined_configuration
from neural_compressor.ux.components.configuration_wizard.params_feeder import get_possible_values
from neural_compressor.ux.components.configuration_wizard.save_workload import save_workload
from neural_compressor.ux.components.file_browser.file_browser import get_directory_entries
from neural_compressor.ux.components.graph.graph import Graph
from neural_compressor.ux.components.graph.graph_reader import GraphReader
from neural_compressor.ux.components.manage_workspace import get_default_path, get_workloads_list, set_workspace
from neural_compressor.ux.components.model_zoo.list_models import list_models
from neural_compressor.ux.components.model_zoo.save_workload import save_workload as save_example_workload
from neural_compressor.ux.components.optimization.execute_optimization import execute_optimization
from neural_compressor.ux.utils.hw_info import HWInfo
from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.web.communication import Request, Response, create_simple_response
from neural_compressor.ux.web.exceptions import ServiceNotFoundException
from neural_compressor.ux.web.service.request_data_processor import RequestDataProcessor
from neural_compressor.ux.web.service.workload import WorkloadService


class RoutingDefinition:
    """Abstract routing definition."""

    def __init__(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Initialize object."""
        self.callback = callback


class RealtimeRoutingDefinition(RoutingDefinition):
    """Routing executed in realtime."""

    pass


class DeferredRoutingDefinition(RoutingDefinition):
    """Routing executed in separate thread."""

    pass


class Router:
    """Connector between api.py and components."""

    def __init__(self) -> None:
        """Initialize object."""
        self.routes: Dict[str, RoutingDefinition] = {
            "filesystem": RealtimeRoutingDefinition(get_directory_entries),
            "save_workload": RealtimeRoutingDefinition(save_workload),
            "configuration": RealtimeRoutingDefinition(get_predefined_configuration),
            "optimize": DeferredRoutingDefinition(_execute_optimization_benchmark),
            "benchmark": DeferredRoutingDefinition(execute_benchmark),
            "get_default_path": RealtimeRoutingDefinition(get_default_path),
            "set_workspace": RealtimeRoutingDefinition(set_workspace),
            "get_workloads_list": RealtimeRoutingDefinition(get_workloads_list),
            "get_boundary_nodes": DeferredRoutingDefinition(get_boundary_nodes),
            "get_possible_values": RealtimeRoutingDefinition(get_possible_values),
            "list_model_zoo": RealtimeRoutingDefinition(list_models),
            "model_graph": RealtimeRoutingDefinition(get_model_graph),
            "system_info": RealtimeRoutingDefinition(get_system_info),
            "workload/config.yaml": RealtimeRoutingDefinition(WorkloadService.get_config),
            "workload/output.log": RealtimeRoutingDefinition(WorkloadService.get_output),
            "workload/code_template.py": RealtimeRoutingDefinition(
                WorkloadService.get_code_template,
            ),
            "workload/tuning_history": DeferredRoutingDefinition(
                WorkloadService.request_history_snapshot,
            ),
            "save_example_workload": DeferredRoutingDefinition(save_example_workload),
        }

    def handle(self, request: Request) -> Response:
        """Run operation on requested component and return result."""
        routing_definition = self.routes.get(request.operation)
        if routing_definition is None:
            raise ServiceNotFoundException(f"Unable to find {request.operation}")

        data = self._process_routing_definition(routing_definition, request.data)
        if isinstance(data, WebResponse):
            return data

        serialized_data = JsonSerializer.serialize_item(data)

        return create_simple_response(serialized_data)

    @staticmethod
    def _process_routing_definition(
        routing_definition: RoutingDefinition,
        data: dict,
    ) -> Any:
        """Process routing definition."""
        if isinstance(routing_definition, RealtimeRoutingDefinition):
            return routing_definition.callback(data)
        if isinstance(routing_definition, DeferredRoutingDefinition):
            t = Thread(target=routing_definition.callback, args=(data,))
            t.daemon = True
            t.start()
            return {"exit_code": 102, "message": "processing"}
        raise ValueError(
            f"Unsupported RoutingDefinition type: {routing_definition.__class__.__name__}",
        )


def _execute_optimization_benchmark(data: dict) -> None:
    """Execute both tuning and benchmark."""
    optimization_data = execute_optimization(data)
    models_info = optimization_data.get("execution_details", {}).get("optimization", {})
    benchmark_data = {
        "id": data.get("id"),
        "input_model": {
            "precision": models_info.get("input_precision"),
            "path": models_info.get("input_graph"),
        },
        "optimized_model": {
            "precision": models_info.get("output_precision"),
            "path": models_info.get("output_graph"),
        },
    }
    if not optimization_data.get("is_custom_dataloader", None):
        execute_benchmark(benchmark_data)


def get_model_graph(data: Dict[str, Any]) -> Graph:
    """Get model graph."""
    graph_reader = GraphReader()
    return graph_reader.read(
        model_path=RequestDataProcessor.get_string_value(data, "path"),
        expanded_groups=data.get("group", []),
    )


def get_system_info(data: Dict[str, Any]) -> dict:
    """Get system info."""
    hw_info = HWInfo().serialize()
    if "cores" in hw_info:
        del hw_info["cores"]
    return hw_info
