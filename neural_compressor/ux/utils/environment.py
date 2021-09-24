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
"""Environment manager class."""
import os
import sys

from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.utils.workload.workload import WorkloadMigrator
from neural_compressor.ux.utils.workload.workloads_list import WorkloadsListMigrator


class Environment:
    """Environment manager class."""

    @staticmethod
    def ensure_workdir_exists_and_writeable() -> None:
        """Ensure that configured directory exists and can be used."""
        from neural_compressor.ux.utils.logger import log
        from neural_compressor.ux.web.configuration import Configuration

        configuration = Configuration()
        workdir = configuration.workdir
        error_message_tail = "Please ensure it is a directory that can be written to.\nExiting.\n"
        try:
            os.makedirs(workdir, exist_ok=True)
        except Exception as e:
            print(f"Unable to create workdir at {workdir}: {e}.\n{error_message_tail}")
            log.error(e)
            sys.exit(1)
        if not os.access(workdir, os.W_OK):
            print(f"Unable to create files at {workdir}.\n{error_message_tail}")
            sys.exit(2)

    @staticmethod
    def migrate_workloads_list() -> None:
        """Migrate workloads list to latest format version."""
        workload_list_migrator = WorkloadsListMigrator()
        if workload_list_migrator.require_migration:
            workload_list_migrator.migrate()
            workload_list_migrator.dump()

    @staticmethod
    def migrate_workloads() -> None:
        """Migrate workloads to latest format version."""
        workload_list_migrator = WorkloadsListMigrator()
        if not os.path.isfile(workload_list_migrator.workloads_json):
            return
        workload_list_migrator.load_workloads_data()
        updated_workloads = {}
        for workload_id, workload_data in workload_list_migrator.workloads_data.get(
            "workloads",
            {},
        ).items():
            try:
                workload_path = workload_data.get("workload_path", None)
                if workload_path is None:
                    continue
                workload_json_path = os.path.join(workload_path, "workload.json")
                workload_migrator = WorkloadMigrator(
                    workload_json_path=workload_json_path,
                )
                workload_migrator.migrate()
                workload_migrator.dump()
                updated_workloads[workload_id] = workload_data
            except Exception:
                pass
        workload_list_migrator.workloads_data["workloads"] = updated_workloads
        workload_list_migrator.dump()

    @staticmethod
    def clean_workloads_wip_status() -> None:
        """Clean WIP status for workloads in workloads_list.json."""
        workdir = Workdir()
        workdir.clean_status(status_to_clean="wip")
