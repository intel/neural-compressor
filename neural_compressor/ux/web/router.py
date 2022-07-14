# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022 Intel Corporation
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
from neural_compressor.ux.components.configuration_wizard.get_boundary_nodes import (
    get_boundary_nodes,
)
from neural_compressor.ux.components.configuration_wizard.params_feeder import get_possible_values
from neural_compressor.ux.components.db_manager.db_operations import (
    BenchmarkAPIInterface,
    DatasetAPIInterface,
    DiagnosisAPIInterface,
    DictionariesAPIInterface,
    ExamplesAPIInterface,
    ModelAPIInterface,
    OptimizationAPIInterface,
    ProfilingAPIInterface,
    ProjectAPIInterface,
)
from neural_compressor.ux.components.file_browser.file_browser import get_directory_entries
from neural_compressor.ux.components.graph.graph import Graph
from neural_compressor.ux.components.graph.graph_reader import GraphReader
from neural_compressor.ux.components.manage_workspace import get_default_path
from neural_compressor.ux.components.model_zoo.list_models import list_models
from neural_compressor.ux.components.optimization.execute_optimization import execute_optimization
from neural_compressor.ux.components.profiling.execute_profiling import execute_profiling
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.hw_info import HWInfo
from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.web.communication import Request, Response, create_simple_response
from neural_compressor.ux.web.exceptions import ServiceNotFoundException
from neural_compressor.ux.web.service.benchmark import BenchmarkService
from neural_compressor.ux.web.service.model import ModelService
from neural_compressor.ux.web.service.optimization import OptimizationService
from neural_compressor.ux.web.service.profiling import ProfilingService
from neural_compressor.ux.web.service.request_data_processor import RequestDataProcessor


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
            "get_default_path": RealtimeRoutingDefinition(get_default_path),
            "get_possible_values": RealtimeRoutingDefinition(get_possible_values),
            "list_model_zoo": RealtimeRoutingDefinition(list_models),
            "system_info": RealtimeRoutingDefinition(get_system_info),
            "project": RealtimeRoutingDefinition(ProjectAPIInterface.get_project_details),
            "project/create": RealtimeRoutingDefinition(ProjectAPIInterface.create_project),
            "project/delete": RealtimeRoutingDefinition(ProjectAPIInterface.delete_project),
            "project/list": RealtimeRoutingDefinition(ProjectAPIInterface.list_projects),
            "project/note": RealtimeRoutingDefinition(ProjectAPIInterface.update_project_notes),
            "dataset": RealtimeRoutingDefinition(DatasetAPIInterface.get_dataset_details),
            "dataset/add": RealtimeRoutingDefinition(DatasetAPIInterface.add_dataset),
            "dataset/delete": RealtimeRoutingDefinition(DatasetAPIInterface.delete_dataset),
            "dataset/list": RealtimeRoutingDefinition(DatasetAPIInterface.list_datasets),
            "dataset/predefined": RealtimeRoutingDefinition(
                DatasetAPIInterface.get_predefined_dataset,
            ),
            "optimization": RealtimeRoutingDefinition(
                OptimizationAPIInterface.get_optimization_details,
            ),
            "optimization/add": RealtimeRoutingDefinition(
                OptimizationAPIInterface.add_optimization,
            ),
            "optimization/delete": RealtimeRoutingDefinition(
                OptimizationAPIInterface.delete_optimization,
            ),
            "optimization/list": RealtimeRoutingDefinition(
                OptimizationAPIInterface.list_optimizations,
            ),
            "optimization/execute": DeferredRoutingDefinition(execute_optimization),
            "optimization/config.yaml": RealtimeRoutingDefinition(OptimizationService.get_config),
            "optimization/output.log": RealtimeRoutingDefinition(OptimizationService.get_output),
            "optimization/pin_accuracy_benchmark": RealtimeRoutingDefinition(
                OptimizationAPIInterface.pin_accuracy_benchmark,
            ),
            "optimization/pin_performance_benchmark": RealtimeRoutingDefinition(
                OptimizationAPIInterface.pin_performance_benchmark,
            ),
            "benchmark": RealtimeRoutingDefinition(BenchmarkAPIInterface.get_benchmark_details),
            "benchmark/delete": RealtimeRoutingDefinition(BenchmarkAPIInterface.delete_benchmark),
            "benchmark/add": RealtimeRoutingDefinition(BenchmarkAPIInterface.add_benchmark),
            "benchmark/list": RealtimeRoutingDefinition(BenchmarkAPIInterface.list_benchmarks),
            "benchmark/execute": DeferredRoutingDefinition(execute_benchmark),
            "benchmark/config.yaml": RealtimeRoutingDefinition(BenchmarkService.get_config),
            "benchmark/output.log": RealtimeRoutingDefinition(BenchmarkService.get_output),
            "profiling": RealtimeRoutingDefinition(ProfilingAPIInterface.get_profiling_details),
            "profiling/delete": RealtimeRoutingDefinition(ProfilingAPIInterface.delete_profiling),
            "profiling/results/csv": RealtimeRoutingDefinition(ProfilingService.generate_csv),
            "profiling/add": RealtimeRoutingDefinition(ProfilingAPIInterface.add_profiling),
            "profiling/list": RealtimeRoutingDefinition(ProfilingAPIInterface.list_profilings),
            "profiling/execute": DeferredRoutingDefinition(execute_profiling),
            "profiling/config.yaml": RealtimeRoutingDefinition(ProfilingService.get_config),
            "profiling/output.log": RealtimeRoutingDefinition(ProfilingService.get_output),
            "examples/list": RealtimeRoutingDefinition(list_models),
            "examples/add": DeferredRoutingDefinition(ExamplesAPIInterface.create_project),
            "model/list": RealtimeRoutingDefinition(ModelAPIInterface.list_models),
            "model/boundary_nodes": DeferredRoutingDefinition(get_boundary_nodes),
            "model/graph": RealtimeRoutingDefinition(get_model_graph),
            "model/graph/highlight_pattern": RealtimeRoutingDefinition(find_pattern_in_graph),
            "model/download": RealtimeRoutingDefinition(ModelService.get_model),
            "dict/domains": RealtimeRoutingDefinition(DictionariesAPIInterface.list_domains),
            "dict/domain_flavours": RealtimeRoutingDefinition(
                DictionariesAPIInterface.list_domain_flavours,
            ),
            "dict/optimization_types": RealtimeRoutingDefinition(
                DictionariesAPIInterface.list_optimization_types,
            ),
            "dict/optimization_types/precision": RealtimeRoutingDefinition(
                DictionariesAPIInterface.list_optimization_types_for_precision,
            ),
            "dict/precisions": RealtimeRoutingDefinition(DictionariesAPIInterface.list_precisions),
            "dict/dataloaders": RealtimeRoutingDefinition(
                DictionariesAPIInterface.list_dataloaders,
            ),
            "dict/dataloaders/framework": RealtimeRoutingDefinition(
                DictionariesAPIInterface.list_dataloaders_by_framework,
            ),
            "dict/transforms": RealtimeRoutingDefinition(
                DictionariesAPIInterface.list_transforms,
            ),
            "dict/transforms/framework": RealtimeRoutingDefinition(
                DictionariesAPIInterface.list_transforms_by_framework,
            ),
            "dict/transforms/domain": RealtimeRoutingDefinition(
                DictionariesAPIInterface.list_transforms_by_domain,
            ),
            "dict/metrics": RealtimeRoutingDefinition(DictionariesAPIInterface.list_metrics),
            "dict/metrics/framework": RealtimeRoutingDefinition(
                DictionariesAPIInterface.list_metrics_by_framework,
            ),
            "diagnosis/op_list": RealtimeRoutingDefinition(DiagnosisAPIInterface.get_op_list),
            "diagnosis/op_details": RealtimeRoutingDefinition(
                DiagnosisAPIInterface.get_op_details,
            ),
            "diagnosis/histogram": RealtimeRoutingDefinition(
                DiagnosisAPIInterface.histogram,
            ),
            "diagnosis/generate_optimization": RealtimeRoutingDefinition(
                DiagnosisAPIInterface.generate_optimization,
            ),
            "diagnosis/model_wise_params": RealtimeRoutingDefinition(
                DiagnosisAPIInterface.model_wise_params,
            ),
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

    def _process_routing_definition(
        self,
        routing_definition: RoutingDefinition,
        data: dict,
    ) -> Any:
        """Process routing definition."""
        if isinstance(routing_definition, RealtimeRoutingDefinition):
            return routing_definition.callback(data)
        if isinstance(routing_definition, DeferredRoutingDefinition):
            self._validate_deffered_routing_data(data)
            t = Thread(target=routing_definition.callback, args=(data,))
            t.daemon = True
            t.start()
            return {"exit_code": 102, "message": "processing"}
        raise ValueError(
            f"Unsupported RoutingDefinition type: {routing_definition.__class__.__name__}",
        )

    @staticmethod
    def _validate_deffered_routing_data(data: dict) -> None:
        """Validate input data for Deffered Routing and raises in case of issues."""
        request_id = str(data.get("request_id", ""))
        if not request_id:
            raise ClientErrorException("Missing request id.")


def get_model_graph(data: Dict[str, Any]) -> Graph:
    """Get model graph."""
    graph_reader = GraphReader()
    return graph_reader.read(
        model_path=RequestDataProcessor.get_string_value(data, "path"),
        expanded_groups=data.get("group", []),
    )


def find_pattern_in_graph(data: Dict[str, Any]) -> dict:
    """Find OP pattern in graph for diagnosis tab."""
    graph_reader = GraphReader()

    model_path = RequestDataProcessor.get_string_value(data, "path")
    op_name = data.get("op_name", None)
    pattern = data.get("pattern", None)
    if any([param is None for param in [model_path, op_name, pattern]]):
        raise ClientErrorException(
            "Missing parameters. Required parameters are: path, op_name and pattern.",
        )
    model_graph, expanded_groups = graph_reader.find_pattern_in_graph(
        model_path=model_path,
        op_name=op_name,
        pattern=pattern,
    )
    return {"graph": model_graph.serialize(), "groups": expanded_groups}


def get_system_info(data: Dict[str, Any]) -> dict:
    """Get system info."""
    hw_info = HWInfo().serialize()
    if "cores" in hw_info:
        del hw_info["cores"]
    return hw_info
