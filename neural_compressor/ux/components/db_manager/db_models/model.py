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
"""The Model class."""
import json
from typing import Any, List, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship, session
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base, DBManager
from neural_compressor.ux.utils.logger import log

log.debug("Initializing Model table")

db_manager = DBManager()


class Model(Base):
    """INC Bench models' table representation."""

    __tablename__ = "model"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("project.id"))
    name = Column(String(50), nullable=False)
    path = Column(String(250), nullable=False)
    framework_id = Column(Integer, ForeignKey("framework.id"), nullable=False)
    size = Column(Float, nullable=False)
    precision_id = Column(Integer, ForeignKey("precision.id"), nullable=False)
    domain_id = Column(Integer, ForeignKey("domain.id"), nullable=False)
    domain_flavour_id = Column(Integer, ForeignKey("domain_flavour.id"), nullable=False)
    input_nodes = Column(String(250), nullable=False, default="")
    output_nodes = Column(String(250), nullable=False, default="")
    supports_profiling = Column(Boolean, nullable=False, default=False)
    supports_graph = Column(Boolean, nullable=False, default=False)
    supports_pruning = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=func.now())

    project: Any = relationship("Project", back_populates="models")

    optimization: Any = relationship(
        "Optimization",
        back_populates="optimized_model",
        primaryjoin="Optimization.optimized_model_id == Model.id",
    )

    benchmarks: Any = relationship(
        "Benchmark",
        back_populates="model",
        cascade="all, delete",
    )

    profilings: Any = relationship(
        "Profiling",
        back_populates="model",
        cascade="all, delete",
    )

    framework: Any = relationship(
        "Framework",
        back_populates="models",
        primaryjoin="Framework.id == Model.framework_id",
    )
    precision: Any = relationship(
        "Precision",
        back_populates="models",
        primaryjoin="Precision.id == Model.precision_id",
    )
    domain: Any = relationship(
        "Domain",
        back_populates="models",
        primaryjoin="Domain.id == Model.domain_id",
    )
    domain_flavour: Any = relationship(
        "DomainFlavour",
        back_populates="models",
        primaryjoin="DomainFlavour.id == Model.domain_flavour_id",
    )

    @staticmethod
    def add(
        db_session: session.Session,
        project_id: int,
        name: str,
        path: str,
        input_nodes: List[str],
        output_nodes: List[str],
        size: float,
        framework_id: int,
        precision_id: int,
        domain_id: int,
        domain_flavour_id: int,
        supports_profiling: bool,
        supports_graph: bool,
        supports_pruning: bool,
    ) -> int:
        """
        Add model to database.

        returns id of added model
        """
        new_model = Model(
            project_id=project_id,
            name=name,
            path=path,
            input_nodes=json.dumps(input_nodes),
            output_nodes=json.dumps(output_nodes),
            size=size,
            framework_id=framework_id,
            precision_id=precision_id,
            domain_id=domain_id,
            domain_flavour_id=domain_flavour_id,
            supports_profiling=supports_profiling,
            supports_graph=supports_graph,
            supports_pruning=supports_pruning,
        )
        db_session.add(new_model)
        db_session.flush()

        return int(new_model.id)

    @staticmethod
    def details(db_session: session.Session, model_id: int) -> dict:
        """Get model details."""
        model = db_session.query(Model).filter(Model.id == model_id)[0]
        return Model.build_info(model)

    @staticmethod
    def list(db_session: session.Session, project_id: int) -> dict:
        """Get models list for specified project from database."""
        models = []
        model_instances = db_session.query(
            Model.id,
            Model.name,
            Model.path,
            Model.precision_id,
            Model.created_at,
        ).filter(Model.project_id == project_id)
        for model in model_instances:
            models.append(
                {
                    "id": model.id,
                    "name": model.name,
                    "path": model.path,
                    "precision_id": model.precision_id,
                    "created_at": str(model.created_at),
                },
            )
        return {"models": models}

    @staticmethod
    def delete_model(
        db_session: session.Session,
        model_id: int,
        model_name: str,
    ) -> Optional[int]:
        """Remove model from database."""
        model = (
            db_session.query(Model)
            .filter(Model.id == model_id)
            .filter(Model.name == model_name)
            .one_or_none()
        )
        if model is None:
            return None
        db_session.delete(model)
        db_session.flush()

        return int(model.id)

    @staticmethod
    def build_info(model: Any) -> dict:
        """Get model info."""
        return {
            "id": model.id,
            "name": model.name,
            "path": model.path,
            "framework": {
                "id": model.framework.id,
                "name": model.framework.name,
            },
            "size": model.size,
            "precision": {
                "id": model.precision.id,
                "name": model.precision.name,
            },
            "domain": {
                "id": model.domain.id,
                "name": model.domain.name,
            },
            "domain_flavour": {
                "id": model.domain_flavour.id,
                "name": model.domain_flavour.name,
            },
            "input_nodes": json.loads(model.input_nodes),
            "output_nodes": json.loads(model.output_nodes),
            "supports_profiling": model.supports_profiling,
            "supports_graph": model.supports_graph,
            "supports_pruning": model.supports_pruning,
            "created_at": str(model.created_at),
        }
