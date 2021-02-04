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

import os
from typing import Any, Dict

from lpot.ux.utils.exceptions import ClientErrorException


def get_default_path(data: Dict[str, Any]) -> Dict[str, Any]:
    """Get paths repository or workspace."""
    env_name = data.get("name", None)
    if not env_name:
        raise ClientErrorException("Could not find proper env.")

    map_param_to_env = {
        "lpot_repository": "LPOT_REPOSITORY_PATH",
        "workspace": "HOME",
    }

    return {"path": os.environ[map_param_to_env[env_name]]}


def set_workspace(data: Dict[str, Any]) -> Dict[str, Any]:
    """Set workspace."""
    workspace_path = data.get("path", None)

    if not workspace_path:
        raise ClientErrorException("Parameter 'path' is missing.")

    os.makedirs(workspace_path, exist_ok=True)

    return {"message": "SUCCESS"}
