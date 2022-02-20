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
"""The ProfilingResults class."""
from typing import Any, List

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import relationship, session
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.components.db_manager.params_interfaces import (
    ProfilingResultAddParamsInterface,
)
from neural_compressor.ux.utils.logger import log

log.debug("Initializing ProfilingResult table")


class ProfilingResult(Base):
    """INC Bench profiling results' table representation."""

    __tablename__ = "profiling_result"

    id = Column(Integer, primary_key=True, index=True, unique=True)
    profiling_id = Column(Integer, ForeignKey("profiling.id"))
    node_name = Column(String)
    total_execution_time = Column(Integer, nullable=False)
    accelerator_execution_time = Column(Integer, nullable=False)
    cpu_execution_time = Column(Integer, nullable=False)
    op_run = Column(Integer, nullable=False)
    op_defined = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
    __table_args__ = (UniqueConstraint("profiling_id", "node_name", name="_profiling_node_uc"),)

    profiling: Any = relationship("Profiling", back_populates="results")

    @staticmethod
    def add(
        db_session: session.Session,
        profiling_id: int,
        node_name: str,
        total_execution_time: int,
        accelerator_execution_time: int,
        cpu_execution_time: int,
        op_run: int,
        op_defined: int,
    ) -> None:
        """Add profiling result to database."""
        insert_statement = insert(ProfilingResult).values(
            profiling_id=profiling_id,
            node_name=node_name,
            total_execution_time=total_execution_time,
            accelerator_execution_time=accelerator_execution_time,
            cpu_execution_time=cpu_execution_time,
            op_run=op_run,
            op_defined=op_defined,
        )
        upsert_statement = insert_statement.on_conflict_do_update(
            index_elements=["profiling_id", "node_name"],
            set_={
                "total_execution_time": total_execution_time,
                "accelerator_execution_time": accelerator_execution_time,
                "cpu_execution_time": cpu_execution_time,
                "op_run": op_run,
                "op_defined": op_defined,
            },
        )
        db_session.execute(upsert_statement)
        db_session.flush()

    @staticmethod
    def get_results(
        db_session: session.Session,
        profiling_id: int,
    ) -> dict:
        """Get results for specified profiling."""
        profiling_results: List[dict] = []
        profiling_result_instances = (
            db_session.query(
                ProfilingResult,
            )
            .filter(ProfilingResult.profiling_id == profiling_id)
            .all()
        )
        for result in profiling_result_instances:
            profiling_info = ProfilingResult.build_info(result)
            profiling_results.append(profiling_info)
        return {"profiling_results": profiling_results}

    @staticmethod
    def bulk_add(
        db_session: session.Session,
        profiling_id: int,
        results: List[ProfilingResultAddParamsInterface],
    ) -> None:
        """Bulk add profiling result to database."""
        profiling_results = []
        for result in results:
            profiling_results.append(
                ProfilingResult(
                    profiling_id=profiling_id,
                    node_name=result.node_name,
                    total_execution_time=result.total_execution_time,
                    accelerator_execution_time=result.accelerator_execution_time,
                    cpu_execution_time=result.cpu_execution_time,
                    op_run=result.op_run,
                    op_defined=result.op_defined,
                ),
            )
        db_session.bulk_save_objects(profiling_results)

    @staticmethod
    def delete_results(
        db_session: session.Session,
        profiling_id: int,
    ) -> None:
        """Delete results for specified profiling."""
        db_session.query(ProfilingResult).filter(
            ProfilingResult.profiling_id == profiling_id,
        ).delete(synchronize_session="fetch")

    @staticmethod
    def build_info(
        result: Any,
    ) -> dict:
        """Build profiling result info."""
        return {
            "profiling_id": result.profiling_id,
            "id": result.id,
            "node_name": result.node_name,
            "total_execution_time": result.total_execution_time,
            "accelerator_execution_time": result.accelerator_execution_time,
            "cpu_execution_time": result.cpu_execution_time,
            "op_run": result.op_run,
            "op_defined": result.op_defined,
            "created_at": str(result.created_at),
        }
