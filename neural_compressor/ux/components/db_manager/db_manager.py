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

"""INC Bench database manager."""

import logging
import os
from typing import Any, Optional

from sqlalchemy import MetaData, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base

from neural_compressor.ux.utils.consts import WORKDIR_LOCATION
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.singleton import Singleton

naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

meta = MetaData(naming_convention=naming_convention)
Base: Any = declarative_base(metadata=meta)


class DBManager(metaclass=Singleton):
    """Database manager class."""

    def __init__(self, database_location: Optional[str] = None, log_level: Optional[int] = None):
        """Initialize database manager."""
        self._engine: Optional[Engine] = None
        self.database_location: str = os.path.join(WORKDIR_LOCATION, "bench.db")
        self.debug: bool = False
        self.dialect: str = "sqlite"

        if database_location is not None:
            self.database_location = database_location

        self.database_entrypoint = f"{self.dialect}:///{self.database_location}"

        if log_level == logging.DEBUG:
            self.debug = True

    def initialize_database(self) -> None:
        """Initialize database by creating engine and session."""
        self.create_sqlalchemy_engine()

    def create_sqlalchemy_engine(self) -> Engine:
        """Create SQLAlchemy engine."""
        log.debug(f"Making engine with database: {self.database_entrypoint}")
        return create_engine(self.database_entrypoint, echo=self.debug)

    @property
    def engine(self) -> Engine:
        """Ensure that SQLAlchemy engine is created."""
        is_engine_instance = isinstance(self._engine, Engine)
        if not is_engine_instance:
            self._engine = self.create_sqlalchemy_engine()
        return self._engine  # type: ignore

    def create_all(self) -> None:
        """Make a call to database to create all tables."""
        log.debug("Creating connection")
        connection = self.engine.connect()
        try:
            log.debug("Creating all")
            Base.metadata.create_all(self.engine)
        finally:
            connection.close()
