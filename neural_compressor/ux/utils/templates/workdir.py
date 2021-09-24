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
"""Workdir class."""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from neural_compressor.ux.utils.templates.metric import Metric
from neural_compressor.ux.utils.workload.workloads_list import WorkloadInfo
from neural_compressor.ux.web.configuration import Configuration


class Workdir:
    """Metric represents data which is sent in get_workloads_list endpoint."""

    def __init__(
        self,
        request_id: Optional[str] = None,
        project_name: Optional[str] = None,
        model_path: Optional[str] = None,
        input_precision: Optional[str] = None,
        model_output_path: Optional[str] = None,
        output_precision: Optional[str] = None,
        mode: Optional[str] = None,
        metric: Optional[Union[dict, Metric]] = None,
        overwrite: bool = True,
        created_at: Optional[str] = None,
    ) -> None:
        """Initialize workdir class."""
        configuration = Configuration()
        workspace_path = configuration.workdir
        self.workdir_path = os.path.join(os.environ.get("HOME", ""), ".neural_compressor")
        self.ensure_working_path_exists()
        self.workloads_json = os.path.join(self.workdir_path, "workloads_list.json")
        self.request_id = request_id
        self.workload_path: str
        if not metric:
            metric = Metric()

        if os.path.isfile(self.workloads_json):
            self.workloads_data = self.load()
        else:
            self.workloads_data = {
                "active_workspace_path": workspace_path,
                "workloads": {},
                "version": "3",
            }

        workload_data = self.get_workload_data(request_id)
        if workload_data:
            self.workload_path = workload_data.get("workload_path", "")
        elif workspace_path and request_id:
            workload_name = request_id
            if model_path:
                workload_name = "_".join(
                    [
                        Path(model_path).stem,
                        request_id,
                    ],
                )
            self.workload_path = os.path.join(
                workspace_path,
                "workloads",
                workload_name,
            )

        if request_id and overwrite:
            self.update_data(
                request_id=request_id,
                project_name=project_name,
                model_path=model_path,
                input_precision=input_precision,
                model_output_path=model_output_path,
                output_precision=output_precision,
                mode=mode,
                metric=metric,
                created_at=created_at,
            )

    def load(self) -> dict:
        """Load json with workloads list."""
        logging.info(f"Loading workloads list from {self.workloads_json}")
        with open(self.workloads_json, encoding="utf-8") as workloads_list:
            self.workloads_data = json.load(workloads_list)
            return self.workloads_data

    def dump(self) -> None:
        """Dump workloads information to json."""
        with open(self.workloads_json, "w") as f:
            json.dump(self.workloads_data, f, indent=4)

        logging.info(f"Successfully saved workload to {self.workloads_json}")

    def ensure_working_path_exists(self) -> None:
        """Set workdir if doesn't exist create."""
        os.makedirs(self.workdir_path, exist_ok=True)

    def map_to_response(self) -> dict:
        """Map to response data."""
        data: Dict[str, Any] = {
            "workspace_path": self.get_active_workspace(),
            "workloads_list": [],
        }
        for key, value in self.workloads_data["workloads"].items():
            data["workloads_list"].append(value)
        return data

    def update_data(
        self,
        request_id: str,
        project_name: Optional[str] = None,
        model_path: Optional[str] = None,
        input_precision: Optional[str] = None,
        model_output_path: Optional[str] = None,
        output_precision: Optional[str] = None,
        mode: Optional[str] = None,
        metric: Optional[Union[Dict[str, Any], Metric]] = Metric(),
        status: Optional[str] = None,
        execution_details: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ) -> None:
        """Update data in workloads.list_json."""
        self.load()
        existing_data = self.get_workload_data(request_id)
        if project_name is None:
            project_name = existing_data.get("project_name", None)
        if created_at is None:
            created_at = existing_data.get("created_at", None)
        workload_info = WorkloadInfo(
            workload_path=self.workload_path,
            request_id=request_id,
            project_name=project_name,
            model_path=model_path,
            input_precision=input_precision,
            model_output_path=model_output_path,
            output_precision=output_precision,
            mode=mode,
            metric=metric,
            status=status,
            code_template_path=self.template_path,
            execution_details=execution_details,
            created_at=created_at,
        ).serialize()
        self.workloads_data["workloads"][request_id] = workload_info
        self.dump()

    def set_workload_status(self, request_id: str, status: str) -> None:
        """Set status of a given Workload."""
        self.load()
        existing_data = self.get_workload_data(request_id)
        if not existing_data:
            return
        self.workloads_data["workloads"][request_id]["status"] = status
        self.dump()

    def update_metrics(
        self,
        request_id: Optional[str],
        metric_data: Dict[str, Any],
    ) -> None:
        """Update metric data in workload."""
        self.load()
        self.workloads_data["workloads"][request_id].get("metric", {}).update(
            metric_data,
        )
        self.workloads_data["workloads"][request_id].update(metric_data)
        self.dump()

    def update_execution_details(
        self,
        request_id: Optional[str],
        execution_details: Dict[str, Any],
    ) -> None:
        """Update metric data in workload."""
        self.load()
        self.workloads_data["workloads"][request_id].get(
            "execution_details",
            {},
        ).update(execution_details)
        self.dump()

    def get_active_workspace(self) -> str:
        """Get active workspace."""
        configuration = Configuration()
        path = self.workloads_data.get("active_workspace_path", configuration.workdir)
        return path

    def set_active_workspace(self, workspace_path: str) -> None:
        """Set active workspace."""
        self.workloads_data["active_workspace_path"] = workspace_path
        self.dump()

    def set_code_template_path(self, code_template_path: str) -> None:
        """Set code_template_path."""
        self.workloads_data["workloads"][self.request_id][
            "code_template_path"
        ] = code_template_path
        self.dump()

    def clean_logs(self) -> None:
        """Clean log files."""
        log_files = [os.path.join(self.workload_path, "output.txt")]
        log_files.extend(
            glob.glob(
                os.path.join(self.workload_path, "*.proc"),
            ),
        )
        log_files.extend(
            glob.glob(
                os.path.join(self.workload_path, "*performance_benchmark.txt"),
            ),
        )
        for file in log_files:
            if os.path.exists(file):
                os.remove(file)

    def clean_status(
        self,
        status_to_clean: Optional[str] = None,
        requests_id: Optional[List[str]] = [],
    ) -> None:
        """Clean status for workloads according to passed parameters."""
        for workload_id, workload_params in self.workloads_data["workloads"].items():
            if requests_id and workload_id not in requests_id:
                continue
            self._clean_workload_status(
                workload_params,
                status_to_clean,
            )
        self.dump()

    @staticmethod
    def _clean_workload_status(workload: dict, status: Optional[str]) -> None:
        """Clean specified workload status."""
        workload_status = workload.get("status")
        if status and workload_status != status:
            return
        workload["status"] = ""

    @property
    def template_path(self) -> Optional[str]:
        """Get template_path."""
        return self.get_workload_data(self.request_id).get("code_template_path", None)

    def get_workload_data(self, request_id: Optional[str]) -> dict:
        """Return data of given Workload."""
        return self.workloads_data.get("workloads", {}).get(request_id, {})
