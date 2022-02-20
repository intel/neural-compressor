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
"""Transform package contains transform class representing transform table in database."""
import json
from typing import Any, List

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import Query, relationship, session, sessionmaker
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base, DBManager
from neural_compressor.ux.components.db_manager.db_models.framework import Framework
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import load_transforms_config

log.debug("Initializing Transform table")

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


transform_domain_association = Table(
    "transform_domain_association",
    Base.metadata,
    Column("transform_id", ForeignKey("transform.id")),
    Column("domain_id", ForeignKey("domain.id")),
)


class Transform(Base):
    """INC Bench Transform table class."""

    __tablename__ = "transform"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50))
    help = Column(String(250))
    params = Column(String)
    framework_id = Column(Integer, ForeignKey("framework.id"))
    domain_id = Column(Integer, ForeignKey("framework.id"))
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    framework: Any = relationship(
        "Framework",
        primaryjoin="Framework.id == Transform.framework_id",
    )
    domains: Any = relationship(
        "Domain",
        secondary=transform_domain_association,
        back_populates="transforms",
    )

    __table_args__ = (
        UniqueConstraint(
            "name",
            "framework_id",
            name="_transform_framework_name_uc",
        ),
    )

    @staticmethod
    def list(db_session: session.Session) -> dict:
        """List available transforms."""
        transform_instances = db_session.query(Transform).order_by(Transform.id)
        transforms = Transform.query_to_list(transform_instances)
        return {"transforms": transforms}

    @staticmethod
    def list_by_framework(db_session: session.Session, framework_id: int) -> dict:
        """List available transforms."""
        transform_instances = (
            db_session.query(Transform)
            .order_by(Transform.id)
            .filter(Transform.framework_id == framework_id)
        )
        transforms = Transform.query_to_list(transform_instances)
        return {"transforms": transforms}

    @staticmethod
    def list_by_domain(db_session: session.Session, domain_id: int) -> dict:
        """List available transforms."""
        transform_instances = (
            db_session.query(Transform)
            .order_by(Transform.id)
            .filter(Transform.framework_id == domain_id)
        )
        transforms = Transform.query_to_list(transform_instances)
        return {"transforms": transforms}

    @staticmethod
    def query_to_list(transforms_query: Query) -> List[dict]:
        """Convert query to list."""
        transforms_list: List[dict] = []
        for transform in transforms_query:
            transforms_list.append(
                {
                    "id": transform.id,
                    "name": transform.name,
                    "help": transform.help,
                    "params": json.loads(transform.params),
                    "framework": {
                        "id": transform.framework.id,
                        "name": transform.framework.name,
                    },
                },
            )
        return transforms_list


@event.listens_for(Transform.__table__, "after_create")
def fill_dictionary(*args: list, **kwargs: dict) -> None:
    """Fill dictionary with default values."""
    with Session.begin() as db_session:
        for fw_transform in load_transforms_config():
            framework = fw_transform.get("name")
            if framework is None:
                continue
            framework_id = Framework.get_framework_id(db_session, framework_name=framework)
            if framework_id is None:
                raise InternalException("Could not found framework in database.")
            for transform in fw_transform.get("params", {}):
                db_session.add(
                    Transform(
                        name=transform.get("name"),
                        help=transform.get("help"),
                        params=json.dumps(transform.get("params", [])),
                        framework_id=framework_id,
                    ),
                )
