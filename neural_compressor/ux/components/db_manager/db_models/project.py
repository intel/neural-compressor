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
"""Project package contains project class representing single project."""
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import relationship, session
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.components.db_manager.db_models.benchmark import Benchmark
from neural_compressor.ux.components.db_manager.db_models.dataset import Dataset
from neural_compressor.ux.components.db_manager.db_models.model import Model
from neural_compressor.ux.components.db_manager.db_models.optimization import Optimization
from neural_compressor.ux.components.db_manager.db_models.profiling import Profiling
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.logger import log

log.debug("Initializing Project table")


class Project(Base):
    """INC Bench Project class."""

    __tablename__ = "project"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    notes = Column(String(250), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    # Relationships
    models: Any = relationship(
        "Model",
        order_by=Model.id,
        back_populates="project",
        cascade="all, delete",
    )

    optimizations: Any = relationship(
        "Optimization",
        order_by=Optimization.id,
        back_populates="project",
        cascade="all, delete",
    )

    benchmarks: Any = relationship(
        "Benchmark",
        order_by=Benchmark.id,
        back_populates="project",
        cascade="all, delete",
    )

    profilings: Any = relationship(
        "Profiling",
        order_by=Profiling.id,
        back_populates="project",
        cascade="all, delete",
    )

    datasets: Any = relationship(
        "Dataset",
        order_by=Dataset.id,
        back_populates="project",
        cascade="all, delete",
    )

    @staticmethod
    def create_project(db_session: session.Session, name: str) -> int:
        """
        Create project object.

        returns id of created project
        """
        new_project = Project(name=name)
        db_session.add(new_project)
        db_session.flush()

        return int(new_project.id)

    @staticmethod
    def delete_project(
        db_session: session.Session,
        project_id: int,
        project_name: str,
    ) -> Optional[int]:
        """Remove project from database."""
        project = (
            db_session.query(Project)
            .filter(Project.id == project_id)
            .filter(Project.name == project_name)
            .one_or_none()
        )
        if project is None:
            return None
        db_session.delete(project)
        db_session.flush()

        return int(project.id)

    @staticmethod
    def project_details(db_session: session.Session, project_id: int) -> dict:
        """Get project details from database."""
        (project, models) = (
            db_session.query(
                Project,
                Model,
            )
            .join(Model)
            .filter(Project.id == project_id)[0]
        )

        return {
            "id": project.id,
            "name": project.name,
            "notes": project.notes,
            "input_model": Model.build_info(models),
            "created_at": str(project.created_at),
            "modified_at": str(project.modified_at),
        }

    @staticmethod
    def get_model_by_name(db_session: session.Session, project_id: int, model_name: str) -> dict:
        """Get input model details for specified project."""
        model = (
            db_session.query(Model)
            .filter(
                Model.project_id == project_id,
                Model.name == model_name,
            )
            .one_or_none()
        )
        if model is None:
            raise ClientErrorException("Could not find specified model in the project.")

        return Model.build_info(model)

    @staticmethod
    def list_projects(db_session: session.Session) -> dict:
        """Get project list from database."""
        projects = []
        project_instances = db_session.query(
            Project.id,
            Project.name,
            Project.created_at,
            Project.modified_at,
        )
        for project in project_instances:
            projects.append(
                {
                    "id": project.id,
                    "name": project.name,
                    "created_at": str(project.created_at),
                    "modified_at": str(project.modified_at),
                },
            )
        return {"projects": projects}

    @staticmethod
    def update_notes(db_session: session.Session, project_id: int, notes: str) -> dict:
        """Update project notes."""
        project = db_session.query(Project).filter(Project.id == project_id)[0]
        project.notes = notes
        db_session.add(project)
        db_session.flush()

        return {
            "id": project.id,
            "notes": project.notes,
        }
