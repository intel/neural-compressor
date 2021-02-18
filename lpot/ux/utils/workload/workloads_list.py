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
"""Workloads list class."""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

from lpot.ux.utils.json_serializer import JsonSerializer
from lpot.ux.utils.templates.metric import Metric
from lpot.ux.utils.utils import get_size

logging.basicConfig(level=logging.INFO)


class WorkloadInfo(JsonSerializer):
    """Create template for workload_list entity."""

    def __init__(
        self,
        request_id: Optional[str],
        workload_path: Optional[str],
        model_path: Optional[str],
        model_output_path: Optional[str],
        metric: Optional[Union[Metric, dict]],
        status: Optional[str],
        code_template_path: Optional[str],
        execution_details: Optional[Dict[str, dict]] = None,
    ) -> None:
        """Initialize configuration WorkloadInfo class."""
        super().__init__()
        self._id = request_id
        self._model_path = model_path
        self._model_output_path = model_output_path
        self._workload_path = workload_path
        self._status = status
        self._metric = metric
        self._code_template_path = code_template_path
        self._config_path: Optional[str] = None
        self._log_path: Optional[str] = None
        self._execution_details = execution_details
        if self._workload_path:
            self._config_path = os.path.join(
                self._workload_path,
                "config.yaml",
            )
            self._log_path = os.path.join(self._workload_path, "output.txt")
            if not os.path.isfile(self._log_path):
                os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
                with open(self._log_path, "w") as log_file:
                    log_file.write("Configuration created.\n")
        if self._model_path and self._metric:
            if isinstance(self._metric, dict) and not self._metric.get("size_fp32"):
                self._metric["size_fp32"] = get_size(self._model_path)
            if isinstance(self._metric, Metric) and not self._metric.size_fp32:
                self._metric.insert_data("size_fp32", str(get_size(self._model_path)))

    def insert_data(self, data: dict) -> None:
        """
        Set all available properties from workload_info dict.

        param: data
        type: dict
        """
        for key, value in data:
            attribute = "_" + key
            if attribute in self.__dict__:
                self.__setattr__(attribute, data[key])

    def serialize(
        self,
        serialization_type: str = "default",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Serialize class to dict.

        :param serialization_type: serialization type, defaults to "default"
        :type serialization_type: str, optional
        :return: serialized class
        :rtype: Union[dict, List[dict]]
        """
        result = {}
        for key, value in self.__dict__.items():
            variable_name = re.sub(r"^_", "", key)
            if key in self._skip:
                continue
            elif issubclass(type(value), JsonSerializer):
                # pylint: disable=maybe-no-member
                result[variable_name] = value.serialize(serialization_type)
            else:
                result[variable_name] = self.serialize_item(value)

        if result.get("metric", None):
            result.update(result["metric"])
        return result
