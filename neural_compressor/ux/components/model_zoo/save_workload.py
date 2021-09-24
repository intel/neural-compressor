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
"""Download config, model and save example workload."""

import logging
import os
from typing import Any, Dict

from neural_compressor.ux.components.configuration_wizard.configuration_parser import ConfigurationParser
from neural_compressor.ux.components.model_zoo.download_config import download_config
from neural_compressor.ux.components.model_zoo.download_model import download_model
from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.utils.workload.workload import Workload
from neural_compressor.ux.web.communication import MessageQueue

logging.basicConfig(level=logging.INFO)


def save_workload(data: Dict[str, Any]) -> None:
    """Get configuration."""
    mq = MessageQueue()

    parser = ConfigurationParser()

    request_id: str = str(data.get("id", ""))
    try:
        config_path = download_config(data)
        model_path = download_model(data)
    except Exception as e:
        mq.post_error(
            "download_finish",
            {"message": str(e), "code": 404, "id": request_id},
        )
        raise

    data["config_path"] = config_path
    data["model_path"] = model_path
    data["project_name"] = os.path.basename(model_path)

    parsed_data = parser.parse(data)

    workload = Workload(parsed_data)
    workload.dump()

    workdir = Workdir(
        request_id=workload.id,
        project_name=workload.project_name,
        model_path=workload.model_path,
        input_precision=workload.input_precision,
        output_precision=workload.output_precision,
        mode=workload.mode,
        created_at=workload.created_at,
    )

    workload.config.set_model_path(data["model_path"])

    workload.config.dump(os.path.join(workdir.workload_path, workload.config_name))

    mq.post_success("example_workload_saved", {"id": workload.id})
