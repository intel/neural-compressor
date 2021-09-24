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
"""Get path to default repository or workspace."""

from typing import Any, Dict

from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.web.configuration import Configuration


def get_default_path(data: Dict[str, Any]) -> Dict[str, Any]:
    """Get paths repository or workspace."""
    configuration = Configuration()
    return {"path": configuration.workdir}


def set_workspace(data: Dict[str, Any]) -> Dict[str, Any]:
    """Set workspace."""
    return {"message": "SUCCESS"}


def get_workloads_list(data: dict) -> Dict[str, Any]:
    """Return workloads list."""
    workdir = Workdir()

    return workdir.map_to_response()
