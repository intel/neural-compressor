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
"""Configuration to yaml."""

import json
import logging as log
import os
from typing import Any, Dict

from lpot.ux.web.communication import MessageQueue

mq = MessageQueue()
log.basicConfig(level=log.INFO)


def get_boundary_nodes(data: Dict[str, Any]) -> None:
    """Get configuration."""
    from lpot.ux.utils.utils import find_boundary_nodes

    request_id = data.get("id", None)
    model_path = data.get("model_path", None)

    if not (request_id and model_path):
        message = "Missing model path or request id."
        mq.post_error(
            "boundary_nodes_finish",
            {"message": message, "code": 404, "id": request_id},
        )
        raise Exception(message)

    if not os.path.isfile(model_path):
        message = "Could not found model in specified path."
        mq.post_error(
            "boundary_nodes_finish",
            {"message": message, "code": 404, "id": request_id},
        )
        raise Exception(message)

    try:
        mq.post_success(
            "boundary_nodes_start",
            {"message": "started", "id": request_id},
        )
        response_data = find_boundary_nodes(model_path)
        response_data["id"] = request_id
    except Exception as err:
        mq.post_error(
            "boundary_nodes_finish",
            {"message": repr(err), "code": 404, "id": request_id},
        )
        raise err
    log.info(f"Parsed data is {json.dumps(response_data)}")
    mq.post_success("boundary_nodes_finish", response_data)
