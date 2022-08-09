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

from alembic import command
from alembic.config import Config
from sqlalchemy.exc import OperationalError

from neural_compressor.ux.components.db_manager import DBManager
from neural_compressor.ux.utils.consts import ExecutionStatus
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.templates.workdir import Workdir

db_manager = DBManager()


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
            log.error(f"Unable to create workdir at {workdir}: {e}.\n{error_message_tail}")
            log.error(e)
            sys.exit(1)
        if not os.access(workdir, os.W_OK):
            log.error(f"Unable to create files at {workdir}.\n{error_message_tail}")
            sys.exit(2)

    @staticmethod
    def clean_workloads_wip_status() -> None:
        """Clean WIP status for workloads in workloads_list.json."""
        try:
            workdir = Workdir()
            workdir.clean_status(status_to_clean=ExecutionStatus.WIP)
        except OperationalError:
            log.debug("Could not clean WIP status.")

    @staticmethod
    def migrate_database() -> None:
        """Perform database migration to latest version using alembic."""
        if not os.path.isfile(db_manager.database_location):
            log.debug("No database found. Skipping migration.")
            return

        alembic_config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "components",
            "db_manager",
            "alembic.ini",
        )
        alembic_scripts_location = os.path.join(
            os.path.dirname(alembic_config_path),
            "alembic",
        )
        alembic_cfg = Config(alembic_config_path)
        alembic_cfg.set_main_option("sqlalchemy.url", db_manager.database_entrypoint)
        alembic_cfg.set_main_option("script_location", alembic_scripts_location)
        log.debug("Executing DB migration...")
        command.upgrade(alembic_cfg, "head")
