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
"""The Benchmark class."""
from typing import Any, List, Optional

from sqlalchemy import DDL, Column, DateTime, ForeignKey, Integer, String, event
from sqlalchemy.orm import relationship, session
from sqlalchemy.sql import func

from neural_compressor.ux.components.benchmark import Benchmarks
from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.components.db_manager.db_models.benchmark_result import BenchmarkResult
from neural_compressor.ux.components.db_manager.db_models.dataset import Dataset
from neural_compressor.ux.components.db_manager.db_models.model import Model
from neural_compressor.ux.utils.consts import ExecutionStatus
from neural_compressor.ux.utils.logger import log

log.debug("Initializing Benchmark table")


class Benchmark(Base):
    """INC Bench benchmarks' table representation."""

    __tablename__ = "benchmark"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    project_id = Column(Integer, ForeignKey("project.id"))
    model_id = Column(Integer, ForeignKey("model.id"))
    dataset_id = Column(Integer, ForeignKey("dataset.id"), nullable=False)
    mode = Column(String(50), nullable=False, default=Benchmarks.PERF)
    batch_size = Column(Integer, nullable=False)
    warmup_iterations = Column(Integer, nullable=False)
    iterations = Column(Integer, nullable=False)
    number_of_instance = Column(Integer, nullable=False)
    cores_per_instance = Column(Integer, nullable=False)
    config_path = Column(String, nullable=True)
    execution_command = Column(String, nullable=True)
    log_path = Column(String, nullable=True)
    status = Column(String(50))
    created_at = Column(DateTime, nullable=False, default=func.now())
    last_run_at = Column(DateTime, nullable=True)
    duration = Column(Integer, nullable=True)

    project: Any = relationship("Project", back_populates="benchmarks")
    model: Any = relationship("Model", foreign_keys=[model_id])
    dataset = relationship("Dataset", foreign_keys=[dataset_id])
    result: Any = relationship(
        "BenchmarkResult",
        back_populates="benchmark",
        cascade="all, delete",
    )

    @staticmethod
    def add(
        db_session: session.Session,
        name: str,
        project_id: int,
        model_id: int,
        dataset_id: int,
        mode: str,
        batch_size: int,
        iterations: int,
        warmup_iterations: int,
        number_of_instance: int,
        cores_per_instance: int,
        execution_command: str,
    ) -> int:
        """
        Add benchark to database.

        returns id of added benchmark
        """
        new_benchmark = Benchmark(
            name=name,
            project_id=project_id,
            model_id=model_id,
            dataset_id=dataset_id,
            mode=mode,
            batch_size=batch_size,
            iterations=iterations,
            number_of_instance=number_of_instance,
            cores_per_instance=cores_per_instance,
            warmup_iterations=warmup_iterations,
            execution_command=execution_command,
        )
        db_session.add(new_benchmark)
        db_session.flush()

        return int(new_benchmark.id)

    @staticmethod
    def delete_benchmark(
        db_session: session.Session,
        benchmark_id: int,
        benchmark_name: str,
    ) -> Optional[int]:
        """Remove benchmark from database."""
        benchmark = (
            db_session.query(Benchmark)
            .filter(Benchmark.id == benchmark_id)
            .filter(Benchmark.name == benchmark_name)
            .one_or_none()
        )
        if benchmark is None:
            return None
        db_session.delete(benchmark)
        db_session.flush()

        return int(benchmark.id)

    @staticmethod
    def update_status(
        db_session: session.Session,
        benchmark_id: int,
        execution_status: ExecutionStatus,
    ) -> dict:
        """Update benchmark status."""
        benchmark = db_session.query(Benchmark).filter(Benchmark.id == benchmark_id).one()
        benchmark.status = execution_status.value
        db_session.add(benchmark)
        db_session.flush()

        return {
            "id": benchmark.id,
            "status": benchmark.status,
        }

    @staticmethod
    def update_duration(
        db_session: session.Session,
        benchmark_id: int,
        duration: int,
    ) -> dict:
        """Update duration of benchmark."""
        benchmark = db_session.query(Benchmark).filter(Benchmark.id == benchmark_id).one()
        benchmark.duration = duration
        db_session.add(benchmark)
        db_session.flush()

        return {
            "id": benchmark.id,
            "duration": benchmark.duration,
        }

    @staticmethod
    def update_execution_command(
        db_session: session.Session,
        benchmark_id: int,
        execution_command: Optional[str],
    ) -> dict:
        """Update benchmark execution command."""
        benchmark = db_session.query(Benchmark).filter(Benchmark.id == benchmark_id).one()
        benchmark.execution_command = execution_command
        db_session.add(benchmark)
        db_session.flush()

        return {
            "id": benchmark.id,
            "execution_command": benchmark.execution_command,
        }

    @staticmethod
    def update_log_path(
        db_session: session.Session,
        benchmark_id: int,
        path: Optional[str],
    ) -> dict:
        """Update benchmark output log path."""
        benchmark = db_session.query(Benchmark).filter(Benchmark.id == benchmark_id).one()
        benchmark.log_path = path
        db_session.add(benchmark)
        db_session.flush()

        return {
            "id": benchmark.id,
            "log_path": benchmark.log_path,
        }

    @staticmethod
    def update_config_path(
        db_session: session.Session,
        benchmark_id: int,
        path: Optional[str],
    ) -> dict:
        """Update benchmark configuration path."""
        benchmark = db_session.query(Benchmark).filter(Benchmark.id == benchmark_id).one()
        benchmark.config_path = path
        db_session.add(benchmark)
        db_session.flush()

        return {
            "id": benchmark.id,
            "config_path": benchmark.config_path,
        }

    @staticmethod
    def clean_status(
        db_session: session.Session,
        status_to_clean: ExecutionStatus,
    ) -> dict:
        """Clean specified benchmark status from benchmark table."""
        benchmark_ids: List[int] = []
        benchmarks = db_session.query(Benchmark).filter(Benchmark.status == status_to_clean.value)
        for benchmark in benchmarks:
            benchmark.status = None
            benchmark_ids.append(benchmark.id)
            db_session.add(benchmark)
            db_session.flush()

        return {
            "benchmarks_id": benchmark_ids,
        }

    @staticmethod
    def details(db_session: session.Session, benchmark_id: int) -> dict:
        """Get benchmark details."""
        (benchmark, benchmark_result, model, dataset) = (
            db_session.query(
                Benchmark,
                BenchmarkResult,
                Model,
                Dataset,
            )
            .outerjoin(Benchmark.result)
            .join(Benchmark.model)
            .join(Benchmark.dataset)
            .filter(Benchmark.id == benchmark_id)
            .one()
        )

        benchmark_info = Benchmark.build_info(
            benchmark=benchmark,
            result=benchmark_result,
            model=model,
            dataset=dataset,
        )
        return benchmark_info

    @staticmethod
    def list(db_session: session.Session, project_id: int) -> dict:
        """Get benchmarks list for specified project from database."""
        benchmarks = []
        benchmark_instances = (
            db_session.query(
                Benchmark,
                BenchmarkResult,
                Model,
                Dataset,
            )
            .outerjoin(Benchmark.result)
            .join(Benchmark.model)
            .join(Benchmark.dataset)
            .filter(Benchmark.project_id == project_id)
        )
        for benchmark, result, model, dataset in benchmark_instances:
            benchmark_info = Benchmark.build_info(
                benchmark=benchmark,
                result=result,
                model=model,
                dataset=dataset,
            )
            benchmarks.append(benchmark_info)
        return {"benchmarks": benchmarks}

    @staticmethod
    def build_info(
        benchmark: Any,
        result: Optional[BenchmarkResult],
        model: Model,
        dataset: Dataset,
    ) -> dict:
        """Build benchmark info."""
        benchmark_info = {
            "project_id": benchmark.project_id,
            "id": benchmark.id,
            "name": benchmark.name,
            "model": Model.build_info(model),
            "dataset": Dataset.build_info(dataset),
            "mode": benchmark.mode,
            "result": None,
            "batch_size": benchmark.batch_size,
            "warmup_iterations": benchmark.warmup_iterations,
            "iterations": benchmark.iterations,
            "number_of_instance": benchmark.number_of_instance,
            "cores_per_instance": benchmark.cores_per_instance,
            "config_path": benchmark.config_path,
            "log_path": benchmark.log_path,
            "execution_command": benchmark.execution_command,
            "status": benchmark.status,
            "created_at": str(benchmark.created_at),
            "last_run_at": str(benchmark.last_run_at),
            "duration": benchmark.duration,
        }
        if result is not None:
            benchmark_info.update(
                {
                    "result": BenchmarkResult.build_info(result),
                },
            )
        return benchmark_info


update_last_run_date_trigger = """
CREATE TRIGGER update_benchmark_run_date AFTER UPDATE ON benchmark
WHEN NEW.status LIKE "wip"
BEGIN
    UPDATE benchmark SET last_run_at = datetime("now") WHERE id=NEW.id;
END;
"""

event.listen(Benchmark.__table__, "after_create", DDL(update_last_run_date_trigger))
