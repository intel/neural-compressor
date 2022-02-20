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
"""The BenchmarkResult class."""

from typing import Any, Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import relationship, session
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.utils.logger import log

log.debug("Initializing BenchmarkResult table")


class BenchmarkResult(Base):
    """INC Bench benchmarks results' table representation."""

    __tablename__ = "benchmark_result"

    id = Column(Integer, primary_key=True, index=True)
    benchmark_id = Column(Integer, ForeignKey("benchmark.id"), unique=True)
    accuracy = Column(Float, nullable=True)
    performance = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    last_run_at = Column(DateTime, nullable=True)

    benchmark: Any = relationship("Benchmark", back_populates="result")

    @staticmethod
    def add(
        db_session: session.Session,
        benchmark_id: int,
        accuracy: Optional[float],
        performance: Optional[float],
    ) -> None:
        """
        Add benchmark result to database.

        returns id of added result
        """
        insert_statement = insert(BenchmarkResult).values(
            benchmark_id=benchmark_id,
            accuracy=accuracy,
            performance=performance,
        )
        upsert_statement = insert_statement.on_conflict_do_update(
            index_elements=["benchmark_id"],
            set_={
                "accuracy": accuracy,
                "performance": performance,
            },
        )
        db_session.execute(upsert_statement)
        db_session.flush()

    @staticmethod
    def update_accuracy(
        db_session: session.Session,
        benchmark_id: int,
        accuracy: float,
    ) -> dict:
        """
        Update accuracy of existing benchmark result.

        returns dict with benchmark id and changed accuracy
        """
        benchmark_result = (
            db_session.query(BenchmarkResult).filter(BenchmarkResult.id == benchmark_id).one()
        )
        benchmark_result.accuracy = accuracy
        db_session.add(benchmark_result)
        db_session.flush()

        return {
            "id": benchmark_result.id,
            "accuracy": benchmark_result.accuracy,
        }

    @staticmethod
    def update_performance(
        db_session: session.Session,
        benchmark_id: int,
        performance: float,
    ) -> dict:
        """
        Update performance of existing benchmark result.

        returns dict with benchmark id and changed performance
        """
        benchmark_result = (
            db_session.query(BenchmarkResult).filter(BenchmarkResult.id == benchmark_id).one()
        )
        benchmark_result.performance = performance
        db_session.add(benchmark_result)
        db_session.flush()

        return {
            "id": benchmark_result.id,
            "performance": benchmark_result.performance,
        }

    @staticmethod
    def build_info(
        result: Any,
    ) -> dict:
        """Build benchmark result info."""
        return {
            "benchmark_id": result.benchmark.id,
            "id": result.id,
            "accuracy": result.accuracy,
            "performance": result.performance,
            "created_at": str(result.created_at),
            "last_run_at": str(result.last_run_at),
        }
