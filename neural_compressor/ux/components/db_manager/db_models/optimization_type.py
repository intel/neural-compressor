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
"""Optimization_type package contains class representing optimization_type table in database."""
from typing import Any, List

from sqlalchemy import Column, DateTime, Integer, String, event
from sqlalchemy.orm import relationship, session, sessionmaker
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base, DBManager
from neural_compressor.ux.components.db_manager.db_models.precision import (
    Precision,
    precision_optimization_type_association,
)
from neural_compressor.ux.utils.consts import OptimizationTypes
from neural_compressor.ux.utils.logger import log

log.debug("Initializing Optimization_type table")

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class OptimizationType(Base):
    """INC Bench OptimizationType table class."""

    __tablename__ = "optimization_type"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    optimizations: Any = relationship("Optimization", back_populates="optimization_type")
    precisions: Any = relationship(
        "Precision",
        secondary=precision_optimization_type_association,
        back_populates="optimization_types",
    )

    @staticmethod
    def add(db_session: session.Session, name: str) -> int:
        """
        Add optimization type to DB.

        returns id of added optimization type
        """
        new_optimization_type = OptimizationType(name=name)
        db_session.add(new_optimization_type)
        db_session.flush()

        return int(new_optimization_type.id)

    @staticmethod
    def list(db_session: session.Session) -> dict:
        """List available types of optimization."""
        optimization_types = []
        optimization_type_instances = db_session.query(OptimizationType).order_by(
            OptimizationType.id,
        )
        for optimization_type in optimization_type_instances:
            optimization_types.append(
                {
                    "id": optimization_type.id,
                    "name": optimization_type.name,
                },
            )
        return {"optimization_types": optimization_types}

    @staticmethod
    def list_for_precision(db_session: session.Session, precision_name: str) -> dict:
        """List available types of optimization."""
        optimization_types = []
        optimization_type_instances = (
            db_session.query(OptimizationType)
            .select_from(OptimizationType)
            .join(Precision.optimization_types)
            .order_by(OptimizationType.id)
        )
        for optimization_type in optimization_type_instances:
            contains_precision = any(
                [precision_name == precision.name for precision in optimization_type.precisions],
            )
            optimization_types.append(
                {
                    "id": optimization_type.id,
                    "name": optimization_type.name,
                    "is_supported": contains_precision,
                },
            )
        return {"optimization_types": optimization_types}

    @staticmethod
    def get_optimization_type_for_precision(
        db_session: session.Session,
        precision_id: int,
    ) -> List[dict]:
        """Get optimization types for specified precision."""
        optimization_types = []
        optimization_type_instances = (
            db_session.query(OptimizationType)
            .select_from(OptimizationType)
            .join(Precision.optimization_types)
            .order_by(OptimizationType.id)
        )
        for optimization_type in optimization_type_instances:
            contains_precision = any(
                [precision_id == precision.id for precision in optimization_type.precisions],
            )
            if contains_precision:
                optimization_types.append(
                    {
                        "id": optimization_type.id,
                        "name": optimization_type.name,
                    },
                )
        return optimization_types

    @staticmethod
    def get_optimization_type_id(db_session: session.Session, optimization_name: str) -> int:
        """Get optimization type id from name."""
        optimization_type_ids = db_session.query(OptimizationType.id).filter(
            OptimizationType.name == optimization_name,
        )
        if optimization_type_ids.count() != 1:
            raise Exception("Name of optimization type is not unique.")
        return optimization_type_ids[0].id

    @staticmethod
    def get_optimization_type_by_name(db_session: session.Session, optimization_name: str) -> Any:
        """Get optimization type object from name."""
        optimization_type_ids = db_session.query(OptimizationType.id).filter(
            OptimizationType.name == optimization_name,
        )
        if optimization_type_ids.count() != 1:
            raise Exception("Name of optimization type is not unique.")
        return optimization_type_ids[0]

    @staticmethod
    def get_optimization_type_by_id(db_session: session.Session, optimization_id: int) -> Any:
        """Get optimization type object by id."""
        optimization_type_ids = db_session.query(OptimizationType.id).filter(
            OptimizationType.id == optimization_id,
        )
        return optimization_type_ids[0]


@event.listens_for(OptimizationType.__table__, "after_create")
def fill_dictionary(*args: list, **kwargs: dict) -> None:
    """Fill dictionary with default values."""
    with Session.begin() as db_session:
        for optimization_type in OptimizationTypes:
            db_session.add(OptimizationType(name=optimization_type.value))
