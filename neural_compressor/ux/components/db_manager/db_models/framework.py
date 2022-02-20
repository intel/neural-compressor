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
"""Framework package contains framework class representing framework table in database."""
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, event
from sqlalchemy.orm import relationship, session, sessionmaker
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base, DBManager
from neural_compressor.ux.utils.consts import Frameworks
from neural_compressor.ux.utils.logger import log

log.debug("Initializing Framework table")

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class Framework(Base):
    """INC Bench Framework table class."""

    __tablename__ = "framework"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    models: Any = relationship("Model", back_populates="framework")

    @staticmethod
    def get_framework_id(db_session: session.Session, framework_name: str) -> int:
        """Get framework id from name."""
        framework_ids = db_session.query(Framework.id).filter(
            Framework.name.ilike(framework_name),
        )
        if framework_ids.count() < 1:
            raise Exception(f"Framework {framework_name} is not supported.")
        if framework_ids.count() > 1:
            raise Exception("Framework name is ambiguous.")
        return framework_ids[0].id


@event.listens_for(Framework.__table__, "after_create")
def fill_dictionary(*args: list, **kwargs: dict) -> None:
    """Fill dictionary with default values."""
    log.debug("Filling framework dictionary.")
    with Session.begin() as db_session:
        for framework in Frameworks:
            db_session.add(Framework(name=framework.value))
