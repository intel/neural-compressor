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
"""Precision package contains precision class representing precision table in database."""
from typing import Any

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
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.orm.session import Session as SessionType
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base, DBManager
from neural_compressor.ux.utils.consts import Precisions
from neural_compressor.ux.utils.logger import log

log.debug("Initializing Precision table")

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)

precision_optimization_type_association = Table(
    "precision_optimization_type_association",
    Base.metadata,
    Column("optimization_type_id", ForeignKey("optimization_type.id")),
    Column("precision_id", ForeignKey("precision.id")),
    UniqueConstraint(
        "optimization_type_id",
        "precision_id",
        name="_precision_optimziation_type_uc",
    ),
)


class Precision(Base):
    """INC Bench Precision table class."""

    __tablename__ = "precision"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    models: Any = relationship("Model", back_populates="precision")
    optimization_types: Any = relationship(
        "OptimizationType",
        secondary=precision_optimization_type_association,
        back_populates="precisions",
    )

    @staticmethod
    def add(db_session: SessionType, name: str) -> int:
        """
        Add precision to DB.

        returns id of added precision
        """
        new_precision = Precision(name=name)
        db_session.add(new_precision)
        db_session.flush()

        return int(new_precision.id)

    @staticmethod
    def list(db_session: SessionType) -> dict:
        """List available types of optimization."""
        precisions = []
        precision_instances = db_session.query(
            Precision.id,
            Precision.name,
        ).order_by(Precision.id)
        for precision in precision_instances:
            precisions.append(
                {
                    "id": precision.id,
                    "name": precision.name,
                },
            )
        return {"precisions": precisions}

    @staticmethod
    def get_precision_id(db_session: SessionType, precision_name: str) -> int:
        """Get precision id from name."""
        precision_ids = db_session.query(Precision.id).filter(
            Precision.name == precision_name,
        )
        if precision_ids.count() != 1:
            raise Exception("Precision name is not unique.")
        return precision_ids[0].id

    @staticmethod
    def get_precision_by_name(db_session: SessionType, precision_name: str) -> Any:
        """Get precision object from name."""
        precision_ids = db_session.query(Precision.id).filter(
            Precision.name == precision_name,
        )
        if precision_ids.count() != 1:
            raise Exception("Precision name is not unique.")
        return precision_ids[0]

    @staticmethod
    def get_precision_by_id(db_session: SessionType, precision_id: str) -> Any:
        """Get precision object from id."""
        precision_ids = db_session.query(Precision.id).filter(
            Precision.name == precision_id,
        )
        return precision_ids[0]


@event.listens_for(Precision.__table__, "after_create")
def fill_dictionary(*args: list, **kwargs: dict) -> None:
    """Fill dictionary with default values."""
    with Session.begin() as db_session:
        for precision in Precisions:
            db_session.add(Precision(name=precision.value))
