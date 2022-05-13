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

"""Workload service."""

from neural_compressor.ux.components.db_manager.db_operations import ModelAPIInterface
from neural_compressor.ux.utils.exceptions import NotFoundException
from neural_compressor.ux.web.communication import MessageQueue
from neural_compressor.ux.web.service.response_generator import Response, ResponseGenerator

mq = MessageQueue()


class ModelService:
    """Workload related services."""

    @classmethod
    def get_model(cls, data: dict) -> Response:
        """Get config file for requested Workload."""
        model_data = ModelAPIInterface.get_model_details(data)
        model_path = model_data.get("path", None)

        if model_path is None:
            raise NotFoundException("Unable to find model file")

        return ResponseGenerator.serve_from_filesystem(
            path=model_path,
            as_attachment=True,
        )
