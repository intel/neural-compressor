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

from lpot.ux.components.benchmark.execute_benchmark import execute_benchmark
from lpot.ux.components.configuration_wizard.get_boundary_nodes import (
    get_boundary_nodes,
)
from lpot.ux.components.configuration_wizard.get_configuration import (
    get_predefined_configuration,
)
from lpot.ux.components.configuration_wizard.get_predefined_workload import (
    get_predefined_workload,
)
from lpot.ux.components.configuration_wizard.params_feeder import get_possible_values
from lpot.ux.components.file_browser.file_browser import get_directory_entries
from lpot.ux.components.menage_workspace import get_default_path, set_workspace
from lpot.ux.components.model_zoo.download_config import download_config
from lpot.ux.components.model_zoo.download_model import download_model
from lpot.ux.components.model_zoo.list_models import list_models
from lpot.ux.components.tune.execute_tune import execute_tuning
from lpot.ux.web.communication import Request, Response, create_simple_response
from lpot.ux.web.exceptions import ServiceNotFoundException


class Router:
    """Connector between api.py and components."""

    def __init__(self) -> None:
        """Initialize object."""
        pass

    def handle(self, request: Request) -> Response:
        """Run operation on requested component and return result."""
        operation_map = {
            "filesystem": get_directory_entries,
            "save_workload": get_predefined_workload,
            "configuration": get_predefined_configuration,
            "tune": process_request_for_tuning,
            "benchmark": process_request_for_benchmark,
            "get_default_path": get_default_path,
            "set_workspace": set_workspace,
            "get_boundary_nodes": process_request_for_boundary_nodes,
            "get_possible_values": get_possible_values,
            "download_model": process_request_for_model_download,
            "list_model_zoo": list_models,
            "download_config": process_request_for_model_config,
        }

        operation = operation_map.get(request.operation)
        if operation is None:
            raise ServiceNotFoundException(f"Unable to find {request.operation}")

        data = operation(request.data)
        return create_simple_response(data)


def process_request_for_tuning(data: dict) -> dict:
    """Set thread and execute tuning."""
    t = Thread(target=execute_tuning, args=(data,))
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
