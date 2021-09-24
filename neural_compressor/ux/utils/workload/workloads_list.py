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

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.templates.metric import Metric
from neural_compressor.ux.utils.utils import get_size

logging.basicConfig(level=logging.INFO)


class WorkloadInfo(JsonSerializer):
    """Create template for workload_list entity."""

    def __init__(
        self,
        request_id: str,
        project_name: Optional[str],
        workload_path: Optional[str],
        model_path: Optional[str],
        input_precision: Optional[str],
        model_output_path: Optional[str],
        output_precision: Optional[str],
        mode: Optional[str],
        metric: Optional[Union[Metric, dict]],
        status: Optional[str],
        code_template_path: Optional[str],
        created_at: Optional[str],
        execution_details: Optional[Dict[str, dict]] = None,
    ) -> None:
        """Initialize configuration WorkloadInfo class."""
        super().__init__()
        self._id = request_id
        self._project_name = project_name
        self._model_path = model_path
        self._input_precision = input_precision
        self._model_output_path = model_output_path
        self._output_precision = output_precision
        self._mode = mode
        self._workload_path = workload_path
        self._status = status
        self._metric = metric
        self._code_template_path = code_template_path
        self._config_path: Optional[str] = None
        self._log_path: Optional[str] = None
        self._execution_details = execution_details
        self._created_at = created_at
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
            if isinstance(self._metric, dict) and not self._metric.get("size_input_model"):
                self._metric["size_input_model"] = get_size(self._model_path)
            if isinstance(self._metric, Metric) and not self._metric.size_input_model:
                self._metric.insert_data("size_input_model", str(get_size(self._model_path)))

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


class WorkloadsListMigrator:
    """Workloads list migrator."""

    def __init__(self) -> None:
        """Initialize workloads list migrator."""
        self.workloads_json = os.path.join(
            os.environ.get("HOME", ""),
            ".neural_compressor",
            "workloads_list.json",
        )
        self.workloads_data: dict = {}
        self.version_migrators: Dict[int, Any] = {
            2: self._migrate_to_v2,
            3: self._migrate_to_v3,
        }

    @property
    def current_version(self) -> int:
        """Get version of current workloads list."""
        self.ensure_workloads_loaded()
        return int(float(self.workloads_data.get("version", 1)))

    @property
    def require_migration(self) -> bool:
        """Check if workloads list require migration."""
        if not os.path.isfile(self.workloads_json):
            log.debug("Workloads list does not exits.")
            return False
        if self.current_version >= max(self.version_migrators.keys()):
            log.debug("Workloads list already up to date.")
            return False
        return True

    def load_workloads_data(self) -> None:
        """Load workloads data from json."""
        with open(self.workloads_json, encoding="utf-8") as workloads_list:
            self.workloads_data = json.load(workloads_list)

    def ensure_workloads_loaded(self) -> None:
        """Make sure that workloads list is loaded."""
        if not self.workloads_data and os.path.isfile(self.workloads_json):
            self.load_workloads_data()

    def dump(self) -> None:
        """Dump workloads information to json."""
        with open(self.workloads_json, "w") as f:
            json.dump(self.workloads_data, f, indent=4)

    def migrate(self) -> None:
        """Migrate workloads list to latest version."""
        self.ensure_workloads_loaded()
        if not self.require_migration:
            return

        migration_steps = range(self.current_version, max(self.version_migrators.keys()))
        for step in migration_steps:
            migration_version = step + 1
            self._migrate_workloads_list(migration_version)

    def _migrate_workloads_list(self, migration_version: int) -> None:
        """Migrate workloads list one version up."""
        migrate_workloads = self.version_migrators.get(migration_version, None)
        if migrate_workloads is None:
            raise InternalException(
                f"Could not migrate workloads list to version {migration_version}",
            )
        migrate_workloads()

    def _migrate_to_v2(self) -> None:
        """Migrate workloads list from v1 to v2."""
        parsed_workload_data = {
            "active_workspace_path": "",
            "workloads": "",
            "version": 2,
        }
        parsed_workload_data.update(
            {"active_workspace_path": self.workloads_data["active_workspace_path"]},
        )
        parsed_workloads = {}
        for workload_id, workload_data in self.workloads_data["workloads"].items():
            parsed_workloads.update({workload_id: self._parse_workload_to_v2(workload_data)})
        parsed_workload_data.update({"workloads": parsed_workloads})
        self.workloads_data = parsed_workload_data

    def _parse_workload_to_v2(self, workload_data: dict) -> dict:
        """Parse workload from v1 to v2."""
        parsed_workload = {
            "input_precision": "fp32",
            "output_precision": "int8",
            "mode": "tuning",
        }
        for key, value in workload_data.items():
            if key == "metric":
                parsed_key, parsed_value = self._parse_metric_to_v2(value)
            elif key == "execution_details":
                parsed_key, parsed_value = self._parse_exec_details_to_v2(value)
            else:
                parsed_key, parsed_value = self._parse_regular_key_to_v2(key, value)
            parsed_workload.update({parsed_key: parsed_value})

        return parsed_workload

    @staticmethod
    def _parse_metric_to_v2(metric_data: dict) -> Tuple:
        """Parse metric from v1 to v2."""
        metric_mappings = {
            "acc_fp32": "acc_input_model",
            "acc_int8": "acc_optimized_model",
            "tuning_time": "optimization_time",
            "size_fp32": "size_input_model",
            "size_int8": "size_optimized_model",
            "perf_throughput_fp32": "perf_throughput_input_model",
            "perf_throughput_int8": "perf_throughput_optimized_model",
        }
        parsed_metric = {}
        for key, value in metric_data.items():
            parsed_key = metric_mappings.get(key, key)
            parsed_metric.update({parsed_key: value})
        return "metric", parsed_metric

    def _parse_exec_details_to_v2(self, execution_details: Optional[dict]) -> Tuple:
        """Parse metric from v1 to v2."""
        parsed_exec_details = {}
        if execution_details is None:
            return "execution_details", None
        if "tuning" in execution_details:
            parsed_exec_details.update(
                {
                    "optimization": {
                        "input_graph": execution_details["tuning"]["model_path"],
                        "input_precision": "fp32",
                        "output_graph": execution_details["tuning"]["model_output_path"],
                        "output_precision": "int8",
                        "framework": execution_details["tuning"]["framework"],
                        "input_nodes": "N/A",
                        "output_nodes": "N/A",
                        "tune": True,
                        "config_path": execution_details["tuning"]["config_path"],
                        "instances": execution_details["tuning"]["instances"],
                        "cores_per_instance": execution_details["tuning"]["cores_per_instance"],
                        "command": execution_details["tuning"]["command"],
                    },
                },
            )
        if "fp32_benchmark" in execution_details:
            parsed_exec_details.update(
                {
                    "input_model_benchmark": {
                        "performance": self._parse_perf_data_to_v2(
                            execution_details["fp32_benchmark"],
                        ),
                    },
                },
            )
        if "int8_benchmark" in execution_details:
            parsed_exec_details.update(
                {
                    "optimized_model_benchmark": {
                        "performance": self._parse_perf_data_to_v2(
                            execution_details["int8_benchmark"],
                        ),
                    },
                },
            )

        return "execution_details", parsed_exec_details

    @staticmethod
    def _parse_perf_data_to_v2(perf_data: dict) -> dict:
        """Perse performance data from v1 to v2."""
        key_mapping = {"datatype": "precision"}
        parsed_perf_data = {}
        for key, value in perf_data.items():
            parsed_key = key
            if key in key_mapping:
                parsed_key = key_mapping[key]
            parsed_perf_data.update({parsed_key: value})
        return parsed_perf_data

    @staticmethod
    def _parse_regular_key_to_v2(key: str, value: Any) -> Tuple:
        """Parse metric from v1 to v2."""
        key_mapping = {
            "fp32": "input_model",
            "int8": "optimized_model",
            "tuning": "optimization",
            "datatype": "precision",
        }
        for old_key, new_key in key_mapping.items():
            if old_key in key:
                key = key.replace(old_key, new_key)
        return key, value

    def _migrate_to_v3(self) -> None:
        """Migrate workloads list from v2 to v3."""
        for workload_id, workload_data in self.workloads_data["workloads"].items():
            workload_data.update(
                {
                    "project_name": os.path.basename(workload_data.get("model_path", "")),
                    "created_at": "2021-07-15T14:19:18.860579",
                },
            )
        self.workloads_data.update(
            {
                "version": 3,
            },
        )
