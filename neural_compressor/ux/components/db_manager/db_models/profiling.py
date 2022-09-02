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
"""The Profiling class."""
from typing import Any, List, Optional

from sqlalchemy import DDL, Column, DateTime, ForeignKey, Integer, String, event
from sqlalchemy.orm import relationship, session
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.components.db_manager.db_models.dataset import Dataset
from neural_compressor.ux.components.db_manager.db_models.model import Model
from neural_compressor.ux.components.db_manager.db_models.profiling_result import ProfilingResult
from neural_compressor.ux.utils.consts import ExecutionStatus
from neural_compressor.ux.utils.logger import log

log.debug("Initializing Profiling table")


class Profiling(Base):
    """INC Bench profilings' table representation."""

    __tablename__ = "profiling"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    project_id = Column(Integer, ForeignKey("project.id"))
    model_id = Column(Integer, ForeignKey("model.id"))
    dataset_id = Column(Integer, ForeignKey("dataset.id"), nullable=False)
    num_threads = Column(Integer, nullable=False)
    execution_command = Column(String, nullable=True)
    log_path = Column(String, nullable=True)
    status = Column(String(50))
    created_at = Column(DateTime, nullable=False, default=func.now())
    last_run_at = Column(DateTime, nullable=True)
    duration = Column(Integer, nullable=True)

    project: Any = relationship("Project", back_populates="profilings")
    model: Any = relationship("Model", foreign_keys=[model_id])
    dataset = relationship("Dataset", foreign_keys=[dataset_id])
    results: Any = relationship(
        "ProfilingResult",
        order_by=ProfilingResult.id,
        back_populates="profiling",
        cascade="all, delete",
    )

    @staticmethod
    def add(
        db_session: session.Session,
        name: str,
        project_id: int,
        model_id: int,
        dataset_id: int,
        num_threads: int,
    ) -> int:
        """
        Add profiling to database.

        returns id of added profiling
        """
        new_profiling = Profiling(
            name=name,
            project_id=project_id,
            model_id=model_id,
            dataset_id=dataset_id,
            num_threads=num_threads,
        )
        db_session.add(new_profiling)
        db_session.flush()

        return int(new_profiling.id)

    @staticmethod
    def delete_profiling(
        db_session: session.Session,
        profiling_id: int,
        profiling_name: str,
    ) -> Optional[int]:
        """Remove profiling from database."""
        profiling = (
            db_session.query(Profiling)
            .filter(Profiling.id == profiling_id)
            .filter(Profiling.name == profiling_name)
            .one_or_none()
        )
        if profiling is None:
            return None
        db_session.delete(profiling)
        db_session.flush()

        return int(profiling.id)

    @staticmethod
    def update_status(
        db_session: session.Session,
        profiling_id: int,
        execution_status: ExecutionStatus,
    ) -> dict:
        """Update profiling status."""
        profiling = db_session.query(Profiling).filter(Profiling.id == profiling_id).one()
        profiling.status = execution_status.value
        db_session.add(profiling)
        db_session.flush()

        return {
            "id": profiling.id,
            "status": profiling.status,
        }

    @staticmethod
    def update_duration(
        db_session: session.Session,
        profiling_id: int,
        duration: int,
    ) -> dict:
        """Update duration of profiling."""
        profiling = db_session.query(Profiling).filter(Profiling.id == profiling_id).one()
        profiling.duration = duration
        db_session.add(profiling)
        db_session.flush()

        return {
            "id": profiling.id,
            "duration": profiling.duration,
        }

    @staticmethod
    def update_execution_command(
        db_session: session.Session,
        profiling_id: int,
        execution_command: Optional[str],
    ) -> dict:
        """Update profiling execution command."""
        profiling = db_session.query(Profiling).filter(Profiling.id == profiling_id).one()
        profiling.execution_command = execution_command
        db_session.add(profiling)
        db_session.flush()

        return {
            "id": profiling.id,
            "execution_command": profiling.execution_command,
        }

    @staticmethod
    def update_log_path(
        db_session: session.Session,
        profiling_id: int,
        path: Optional[str],
    ) -> dict:
        """Update profiling output log path."""
        profiling = db_session.query(Profiling).filter(Profiling.id == profiling_id).one()
        profiling.log_path = path
        db_session.add(profiling)
        db_session.flush()

        return {
            "id": profiling.id,
            "log_path": profiling.log_path,
        }

    @staticmethod
    def update_dataset(
        db_session: session.Session,
        profiling_id: int,
        dataset_id: int,
    ) -> dict:
        """Update profiling dataset."""
        profiling = db_session.query(Profiling).filter(Profiling.id == profiling_id).one()
        profiling.dataset_id = dataset_id
        db_session.add(profiling)
        db_session.flush()

        return {
            "id": profiling.id,
            "dataset_id": profiling.dataset_id,
        }

    @staticmethod
    def update_num_threads(
        db_session: session.Session,
        profiling_id: int,
        num_threads: int,
    ) -> dict:
        """Update profiling num_threads."""
        profiling = db_session.query(Profiling).filter(Profiling.id == profiling_id).one()
        profiling.num_threads = num_threads
        db_session.add(profiling)
        db_session.flush()

        return {
            "id": profiling.id,
            "num_threads": profiling.num_threads,
        }

    @staticmethod
    def clean_status(
        db_session: session.Session,
        status_to_clean: ExecutionStatus,
    ) -> dict:
        """Clean specified profiling status from profiling table."""
        profiling_ids: List[int] = []
        profilings = db_session.query(Profiling).filter(Profiling.status == status_to_clean.value)
        for profiling in profilings:
            profiling.status = None
            profiling_ids.append(profiling.id)
            db_session.add(profiling)
            db_session.flush()

        return {
            "profilings_id": profiling_ids,
        }

    @staticmethod
    def details(db_session: session.Session, profiling_id: int) -> dict:
        """Get profiling details."""
        (profiling, model, dataset) = (
            db_session.query(
                Profiling,
                Model,
                Dataset,
            )
            .join(Profiling.model)
            .join(Profiling.dataset)
            .filter(Profiling.id == profiling_id)
            .one()
        )

        results = ProfilingResult.get_results(
            db_session=db_session,
            profiling_id=profiling_id,
        )

        profiling_info = Profiling.build_info(
            profiling=profiling,
            results=results.get("profiling_results", []),
            model=model,
            dataset=dataset,
        )
        return profiling_info

    @staticmethod
    def list(db_session: session.Session, project_id: int) -> dict:
        """Get profilings list for specified project from database."""
        profilings: List[dict] = []
        profiling_instances = (
            db_session.query(
                Profiling,
                Model,
                Dataset,
            )
            .join(Profiling.model)
            .join(Profiling.dataset)
            .filter(Profiling.project_id == project_id)
        )
        for profiling, model, dataset in profiling_instances:
            profiling_info = Profiling.build_info(
                profiling=profiling,
                results=None,
                model=model,
                dataset=dataset,
            )
            profilings.append(profiling_info)
        return {"profilings": profilings}

    @staticmethod
    def build_info(
        profiling: Any,
        results: Optional[List[ProfilingResult]],
        model: Model,
        dataset: Dataset,
    ) -> dict:
        """Build profiling info."""
        profiling_info = {
            "project_id": profiling.project_id,
            "id": profiling.id,
            "name": profiling.name,
            "model": Model.build_info(model),
            "dataset": Dataset.build_info(dataset),
            "num_threads": profiling.num_threads,
            "execution_command": profiling.execution_command,
            "log_path": profiling.log_path,
            "status": profiling.status,
            "created_at": str(profiling.created_at),
            "last_run_at": str(profiling.last_run_at),
            "duration": profiling.duration,
        }
        if results:
            profiling_info.update(
                {
                    "results": results,
                },
            )
        return profiling_info


update_last_run_date_trigger = """
    CREATE TRIGGER update_profiling_run_date AFTER UPDATE ON profiling
    WHEN NEW.status LIKE "wip"
    BEGIN
        UPDATE profiling SET last_run_at = datetime("now") WHERE id=NEW.id;
    END;
    """

event.listen(Profiling.__table__, "after_create", DDL(update_last_run_date_trigger))
