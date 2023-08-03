# -*- coding: utf-8 -*-
# Copyright (c) 2023 Intel Corporation
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
import json
import os
from threading import Thread
from typing import Any, Callable, Dict, List, Optional

from werkzeug.wrappers import Response as WebResponse

from neural_insights.components.diagnosis.diagnosis import Diagnosis
from neural_insights.components.diagnosis.factory import DiagnosisFactory
from neural_insights.components.diagnosis.op_details import OpDetails
from neural_insights.components.graph.graph import Graph
from neural_insights.components.graph.graph_reader import GraphReader
from neural_insights.components.workload_manager.workload_manager import WorkloadManager
from neural_insights.utils.exceptions import ClientErrorException
from neural_insights.utils.json_serializer import JsonSerializer
from neural_insights.web.communication import Request, Response, create_simple_response
from neural_insights.web.exceptions import ServiceNotFoundException
from neural_insights.web.service.request_data_processor import RequestDataProcessor


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
            "workloads": RealtimeRoutingDefinition(get_workloads_list),
            "workloads/delete": RealtimeRoutingDefinition(delete_workload),
            "profiling": RealtimeRoutingDefinition(get_profiling_details),
            "model/graph": RealtimeRoutingDefinition(get_model_graph),
            "model/graph/highlight_pattern": RealtimeRoutingDefinition(find_pattern_in_graph),
            "diagnosis/op_list": RealtimeRoutingDefinition(get_op_list),
            "diagnosis/op_details": RealtimeRoutingDefinition(get_op_details),
            "diagnosis/histogram": RealtimeRoutingDefinition(get_histogram),
            "profiling/result": RealtimeRoutingDefinition(get_profiling_details),
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
    if any(param is None for param in [model_path, op_name, pattern]):
        raise ClientErrorException(
            "Missing parameters. Required parameters are: path, op_name and pattern.",
        )
    model_graph, expanded_groups = graph_reader.find_pattern_in_graph(
        model_path=model_path,
        op_name=op_name,
        pattern=pattern,
    )
    return {"graph": model_graph.serialize(), "groups": expanded_groups}


def get_workloads_list(data: Dict[str, Any]) -> dict:
    """Get workloads list."""
    workload_manager = WorkloadManager()

    serialized_workloads = [workload.serialize() for workload in workload_manager.workloads]
    return {
        "workloads": serialized_workloads,
    }


def delete_workload(data: Dict[str, Any]) -> dict:
    """Remove workload from workloads list."""
    workload_id: Optional[str] = data.get("workload_id", None)
    if workload_id is None:
        raise ClientErrorException("Could not find workload ID.")

    removed_id = WorkloadManager().remove_workload(workload_id)

    return {
        "workload_id": removed_id,
    }


def get_diagnosis(workload_id: str) -> Diagnosis:
    """Get diagnosis object for specified workload."""
    workload = WorkloadManager().get_workload(workload_id)
    diagnosis = DiagnosisFactory.get_diagnosis(workload)
    return diagnosis


def get_op_list(data: Dict[str, Any]) -> List[dict]:
    """Get OP list for model."""
    workload_id: Optional[str] = data.get("workload_id", None)
    if workload_id is None:
        raise ClientErrorException("Could not find workload ID.")
    diagnosis = get_diagnosis(workload_id)

    return diagnosis.get_op_list()


def get_op_details(data: Dict[str, Any]) -> dict:
    """Get OP details for specific OP in model."""
    try:
        workload_id: str = str(data.get("workload_id", None))
        op_name: str = str(data.get("op_name", None))
    except ValueError:
        raise ClientErrorException("Incorrect parameter values.")
    except TypeError:
        raise ClientErrorException("Could not find all required parameters.")

    diagnosis = get_diagnosis(workload_id)

    op_details: Optional[OpDetails] = diagnosis.get_op_details(op_name)
    if op_details is None:
        return {}
    return op_details.serialize()


def get_histogram(data: Dict[str, Any]) -> list:
    """Get histogram of specific tensor in model."""
    try:
        workload_id: str = str(data.get("workload_id", None))
        op_name: str = str(data.get("op_name", None))
        histogram_type: str = str(data.get("type", None))
    except ValueError:
        raise ClientErrorException("Incorrect parameter values.")
    except TypeError:
        raise ClientErrorException("Could not find all required parameters.")

    diagnosis = get_diagnosis(workload_id)

    histogram_type_map = {
        "weights": "weight",
        "activation": "activation",
    }

    parsed_histogram_type: Optional[str] = histogram_type_map.get(histogram_type, None)
    if parsed_histogram_type is None:
        raise ClientErrorException(
            f"Histogram type not supported. "
            f"Use one of following: {histogram_type_map.keys()}",
        )

    histogram_data = diagnosis.get_histogram_data(op_name, parsed_histogram_type)
    return histogram_data


def get_profiling_details(data: Dict[str, Any]) -> List[dict]:
    """Get profiling result."""
    workload_id: Optional[str] = data.get("workload_id", None)
    if workload_id is None:
        raise ClientErrorException("Could not find workload ID.")
    workload = WorkloadManager().get_workload(workload_id)
    profiling_data_path = os.path.join(
        workload.workload_location,
        "profiling_data.json",
    )
    with open(profiling_data_path, "r") as json_file:
        profiling_data = json.load(json_file)
    return profiling_data
