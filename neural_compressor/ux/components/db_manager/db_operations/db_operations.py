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
# pylint: disable=no-member
"""INC Bench API interfaces."""
import os
from typing import Any, Dict, List

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from sqlalchemy.orm import sessionmaker

from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.optimization_type import OptimizationType
from neural_compressor.ux.components.db_manager.db_models.precision import (
    Precision,
    precision_optimization_type_association,
)
from neural_compressor.ux.utils.consts import precision_optimization_types
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.logger import log

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


def set_database_version() -> None:
    """Set version_num in alembic_version table."""
    alembic_config_path = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        "alembic.ini",
    )

    alembic_scripts_location = os.path.join(
        os.path.dirname(alembic_config_path),
        "alembic",
    )
    alembic_cfg = AlembicConfig(alembic_config_path)
    alembic_cfg.set_main_option("sqlalchemy.url", db_manager.database_entrypoint)
    alembic_cfg.set_main_option("script_location", alembic_scripts_location)
    command.ensure_version(alembic_cfg)

    script = ScriptDirectory.from_config(alembic_cfg)
    revision = script.get_revision("head")
    latest_revision = revision.revision
    with db_manager.engine.connect() as conn:
        conn.execute(
            f"INSERT INTO alembic_version(version_num) VALUES ('{latest_revision}') "
            f"ON CONFLICT(version_num) DO UPDATE SET version_num='{latest_revision}';",
        )


def initialize_associations() -> None:
    """Initialize association tables in database."""
    initialize_precision_optimization_types_association()


def initialize_precision_optimization_types_association() -> None:
    """Initialize precision and optimization types association table."""
    with Session.begin() as db_session:
        # Check if there is already data in table:
        association_table_select = precision_optimization_type_association.select()
        association_table_data = db_session.execute(association_table_select)
        if len(association_table_data.all()) > 0:
            log.debug("Precision - Optimization Type association table already exists.")
            return

        optimization_types_db = OptimizationType.list(db_session)
        precisions_db = Precision.list(db_session)
        for optimization_type, precisions in precision_optimization_types.items():
            optimization_type_id = search_in_list_of_dict_for_unique_value(
                optimization_types_db["optimization_types"],
                "name",
                optimization_type.value,
            )["id"]

            for precision in precisions:
                precision_id = search_in_list_of_dict_for_unique_value(
                    precisions_db["precisions"],
                    "name",
                    precision.value,
                )["id"]
                query = precision_optimization_type_association.insert().values(
                    precision_id=precision_id,
                    optimization_type_id=optimization_type_id,
                )
                db_session.execute(query)


def search_in_list_of_dict_for_unique_value(
    list_of_dicts: List[Dict[str, Any]],
    parameter: str,
    value: Any,
) -> Dict[str, Any]:
    """Search for dictionaries with specific unique parameter value."""
    search_results: List[dict] = search_in_list_of_dict(list_of_dicts, parameter, value)
    if len(search_results) > 1:
        raise InternalException("Search result is not unique.")
    return search_results[0]


def search_in_list_of_dict(
    list_of_dicts: List[Dict[str, Any]],
    parameter: str,
    value: Any,
) -> List[Dict[str, Any]]:
    """Search for dictionaries with specific parameter value."""
    matching_entries = []
    for entry in list_of_dicts:
        attribute_value = entry.get(parameter)
        if attribute_value == value:
            matching_entries.append(entry)
    return matching_entries
