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
"""The Optimization class."""
import json
from typing import Any, Dict, List, Optional

from sqlalchemy import DDL, Column, DateTime, ForeignKey, Integer, String, event
from sqlalchemy.orm import relationship, session
from sqlalchemy.sql import func

from neural_compressor.ux.components.benchmark import Benchmarks
from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.components.db_manager.db_models.benchmark import Benchmark
from neural_compressor.ux.components.db_manager.db_models.benchmark_result import BenchmarkResult
from neural_compressor.ux.components.db_manager.db_models.dataset import Dataset
from neural_compressor.ux.components.db_manager.db_models.model import Model
from neural_compressor.ux.components.db_manager.db_models.optimization_type import OptimizationType
from neural_compressor.ux.components.db_manager.db_models.precision import Precision
from neural_compressor.ux.components.db_manager.db_models.tuning_details import TuningDetails
from neural_compressor.ux.components.db_manager.db_models.tuning_history import TuningHistory
from neural_compressor.ux.utils.consts import ExecutionStatus
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException


class Optimization(Base):
    """INC Bench optimizations' table representation."""

    __tablename__ = "optimization"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    project_id = Column(Integer, ForeignKey("project.id"))
    precision_id = Column(Integer, ForeignKey("precision.id"), nullable=False)
    optimization_type_id = Column(Integer, ForeignKey("optimization_type.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("dataset.id"), nullable=False)
    tuning_details_id = Column(Integer, ForeignKey("tuning_details.id"), nullable=True)
    batch_size = Column(Integer, nullable=False)
    sampling_size = Column(Integer, nullable=False)
    optimized_model_id = Column(Integer, ForeignKey("model.id"), nullable=True)
    accuracy_benchmark_id = Column(Integer, ForeignKey("benchmark.id"), nullable=True)
    performance_benchmark_id = Column(Integer, ForeignKey("benchmark.id"), nullable=True)
    config_path = Column(String, nullable=True)
    log_path = Column(String, nullable=True)
    execution_command = Column(String, nullable=True)
    status = Column(String(50))
    created_at = Column(DateTime, nullable=False, default=func.now())
    last_run_at = Column(DateTime, nullable=True)
    duration = Column(Integer, nullable=True)
    diagnosis_config = Column(String, nullable=True, default="null")

    project: Any = relationship("Project", back_populates="optimizations")
    precision = relationship("Precision", foreign_keys=[precision_id])
    optimization_type = relationship("OptimizationType", foreign_keys=[optimization_type_id])
    dataset = relationship("Dataset", foreign_keys=[dataset_id])
    tuning_details = relationship(
        "TuningDetails",
        foreign_keys=[tuning_details_id],
        cascade="all, delete",
    )
    tuning_history = relationship(
        "TuningHistory",
        secondary="join(TuningDetails, TuningHistory, "
        "TuningDetails.tuning_history_id == TuningHistory.id)",
        viewonly=True,
    )
    accuracy_benchmark = relationship("Benchmark", foreign_keys=[accuracy_benchmark_id])
    performance_benchmark = relationship("Benchmark", foreign_keys=[performance_benchmark_id])
    optimized_model = relationship(
        "Model",
        foreign_keys=[optimized_model_id],
        back_populates="optimization",
        cascade="all, delete",
    )

    @staticmethod
    def add(
        db_session: session.Session,
        name: str,
        project_id: int,
        precision_id: int,
        optimization_type_id: int,
        dataset_id: int,
        batch_size: int,
        sampling_size: int,
        tuning_details_id: Optional[int] = None,
        diagnosis_config: Optional[dict] = None,
    ) -> int:
        """
        Add optimization to database.

        returns id of added optimization
        """
        new_optimization = Optimization(
            name=name,
            project_id=project_id,
            precision_id=precision_id,
            optimization_type_id=optimization_type_id,
            dataset_id=dataset_id,
            tuning_details_id=tuning_details_id,
            batch_size=batch_size,
            sampling_size=sampling_size,
            diagnosis_config=json.dumps(diagnosis_config),
        )
        db_session.add(new_optimization)
        db_session.flush()

        return int(new_optimization.id)

    @staticmethod
    def delete_optimization(
        db_session: session.Session,
        optimization_id: int,
        optimization_name: str,
    ) -> Optional[int]:
        """Remove optimization from database."""
        optimization = (
            db_session.query(Optimization)
            .filter(Optimization.id == optimization_id)
            .filter(Optimization.name == optimization_name)
            .one_or_none()
        )
        if optimization is None:
            return None
        db_session.delete(optimization)
        db_session.flush()

        return int(optimization.id)

    @staticmethod
    def update_status(
        db_session: session.Session,
        optimization_id: int,
        execution_status: ExecutionStatus,
    ) -> dict:
        """Update status of optimization."""
        optimization = (
            db_session.query(Optimization).filter(Optimization.id == optimization_id).one()
        )
        optimization.status = execution_status.value
        db_session.add(optimization)
        db_session.flush()

        return {
            "id": optimization.id,
            "status": optimization.status,
        }

    @staticmethod
    def update_optimized_model(
        db_session: session.Session,
        optimization_id: int,
        optimized_model_id: int,
    ) -> dict:
        """Update status of optimization."""
        optimization = (
            db_session.query(Optimization).filter(Optimization.id == optimization_id).one()
        )
        optimization.optimized_model_id = optimized_model_id
        db_session.add(optimization)
        db_session.flush()

        return {
            "id": optimization.id,
            "optimized_model_id": optimization.optimized_model_id,
        }

    @staticmethod
    def update_duration(
        db_session: session.Session,
        optimization_id: int,
        duration: int,
    ) -> dict:
        """Update duration of optimization."""
        optimization = (
            db_session.query(Optimization).filter(Optimization.id == optimization_id).one()
        )
        optimization.duration = duration
        db_session.add(optimization)
        db_session.flush()

        return {
            "id": optimization.id,
            "duration": optimization.duration,
        }

    @staticmethod
    def update_execution_command(
        db_session: session.Session,
        optimization_id: int,
        execution_command: Optional[str],
    ) -> dict:
        """Update optimization execution command."""
        optimization = (
            db_session.query(Optimization).filter(Optimization.id == optimization_id).one()
        )
        optimization.execution_command = execution_command
        db_session.add(optimization)
        db_session.flush()

        return {
            "id": optimization.id,
            "execution_command": optimization.execution_command,
        }

    @staticmethod
    def update_log_path(
        db_session: session.Session,
        optimization_id: int,
        path: Optional[str],
    ) -> dict:
        """Update optimization output log path."""
        optimization = (
            db_session.query(Optimization).filter(Optimization.id == optimization_id).one()
        )
        optimization.log_path = path
        db_session.add(optimization)
        db_session.flush()

        return {
            "id": optimization.id,
            "log_path": optimization.log_path,
        }

    @staticmethod
    def update_config_path(
        db_session: session.Session,
        optimization_id: int,
        path: Optional[str],
    ) -> dict:
        """Update optimization configuration path."""
        optimization = (
            db_session.query(Optimization).filter(Optimization.id == optimization_id).one()
        )
        optimization.config_path = path
        db_session.add(optimization)
        db_session.flush()

        return {
            "id": optimization.id,
            "config_path": optimization.config_path,
        }

    @staticmethod
    def clean_status(
        db_session: session.Session,
        status_to_clean: ExecutionStatus,
    ) -> dict:
        """Clean specified optimization status from optimization table."""
        optimization_ids: List[int] = []
        optimizations = db_session.query(Optimization).filter(
            Optimization.status == status_to_clean.value,
        )
        for optimization in optimizations:
            optimization.status = None
            optimization_ids.append(optimization.id)
            db_session.add(optimization)
            db_session.flush()

        return {
            "optimizations_id": optimization_ids,
        }

    @staticmethod
    def pin_accuracy_benchmark(
        db_session: session.Session,
        optimization_id: int,
        benchmark_id: int,
    ) -> dict:
        """Pin accuracy benchmark to optimization."""
        response = Optimization._pin_benchmark(
            db_session=db_session,
            optimization_id=optimization_id,
            benchmark_id=benchmark_id,
            benchmark_type="accuracy",
        )
        return response

    @staticmethod
    def pin_performance_benchmark(
        db_session: session.Session,
        optimization_id: int,
        benchmark_id: int,
    ) -> dict:
        """Pin performance benchmark to optimization."""
        response = Optimization._pin_benchmark(
            db_session=db_session,
            optimization_id=optimization_id,
            benchmark_id=benchmark_id,
            benchmark_type="performance",
        )
        return response

    @staticmethod
    def _pin_benchmark(
        db_session: session.Session,
        optimization_id: int,
        benchmark_id: int,
        benchmark_type: str,
    ) -> dict:
        """Pin benchmark to optimization."""
        optimization = (
            db_session.query(Optimization).filter(Optimization.id == optimization_id).one()
        )

        benchmark = db_session.query(Benchmark).filter(Benchmark.id == benchmark_id).one()
        if optimization.optimized_model_id != benchmark.model_id:
            raise ClientErrorException(
                "Benchmark model does not come from specified optimization.",
            )
        if benchmark.mode != benchmark_type:
            raise ClientErrorException(
                f"Benchmark type {benchmark.mode} cannot be pinned as {benchmark_type}.",
            )
        if benchmark_type == "accuracy":
            optimization.accuracy_benchmark_id = benchmark_id
        elif benchmark_type == "performance":
            optimization.performance_benchmark_id = benchmark_id
        else:
            raise InternalException("Incorrect benchmark type.")
        db_session.add(optimization)
        db_session.flush()
        return {
            "id": optimization.id,
            f"{benchmark_type}_benchmark_id": getattr(
                optimization,
                f"{benchmark_type}_benchmark_id",
            ),
        }

    @staticmethod
    def get_optimization_by_project_and_model(
        db_session: session.Session,
        project_id: int,
        model_id: int,
    ) -> dict:
        """Get optimization details for specific model."""
        (
            optimization,
            precision,
            optimization_type,
            dataset,
            tuning_details,
            tuning_history,
            optimized_model,
        ) = (
            db_session.query(
                Optimization,
                Precision,
                OptimizationType,
                Dataset,
                TuningDetails,
                TuningHistory,
                Model,
            )
            .join(Optimization.precision)
            .join(Optimization.optimization_type)
            .join(Optimization.dataset)
            .outerjoin(Optimization.tuning_details)
            .outerjoin(Optimization.optimized_model)
            .outerjoin(Optimization.tuning_history)
            .filter(Optimization.project_id == project_id)
            .filter(Optimization.optimized_model_id == model_id)
            .one()
        )
        optimization_info = Optimization.build_info(
            optimization=optimization,
            precision=precision,
            optimization_type=optimization_type,
            dataset=dataset,
            tuning_details=tuning_details,
            tuning_history=tuning_history,
            optimized_model=optimized_model,
        )
        return optimization_info

    @staticmethod
    def details(db_session: session.Session, optimization_id: int) -> dict:
        """Get optimization details."""
        (
            optimization,
            precision,
            optimization_type,
            dataset,
            tuning_details,
            tuning_history,
            optimized_model,
        ) = (
            db_session.query(
                Optimization,
                Precision,
                OptimizationType,
                Dataset,
                TuningDetails,
                TuningHistory,
                Model,
            )
            .join(Optimization.precision)
            .join(Optimization.optimization_type)
            .join(Optimization.dataset)
            .outerjoin(Optimization.tuning_details)
            .outerjoin(Optimization.optimized_model)
            .outerjoin(Optimization.tuning_history)
            .filter(Optimization.id == optimization_id)
            .one()
        )

        optimization_info = Optimization.build_info(
            optimization=optimization,
            precision=precision,
            optimization_type=optimization_type,
            dataset=dataset,
            tuning_details=tuning_details,
            tuning_history=tuning_history,
            optimized_model=optimized_model,
        )
        return optimization_info

    @staticmethod
    def list(db_session: session.Session, project_id: int) -> dict:
        """Get optimizations list for specified project from database."""
        optimizations = []
        optimization_instances = (
            db_session.query(
                Optimization,
                Precision,
                OptimizationType,
                Dataset,
                Model,
            )
            .join(Optimization.precision)
            .join(Optimization.optimization_type)
            .join(Optimization.dataset)
            .outerjoin(Optimization.optimized_model)
            .filter(Optimization.project_id == project_id)
        )
        for optimization, precision, optimization_type, dataset, model in optimization_instances:
            optimization_info = Optimization.build_info(
                optimization=optimization,
                precision=precision,
                optimization_type=optimization_type,
                dataset=dataset,
                optimized_model=model,
            )
            optimizations.append(optimization_info)
        return {"optimizations": optimizations}

    @staticmethod
    def build_info(
        optimization: Any,
        precision: Precision,
        optimization_type: OptimizationType,
        dataset: Dataset,
        tuning_details: Optional[TuningDetails] = None,
        tuning_history: Optional[TuningHistory] = None,
        optimized_model: Optional[Model] = None,
    ) -> dict:
        """Get optimization info."""
        diagnosis_config = None
        if optimization.diagnosis_config is not None:
            diagnosis_config = json.loads(optimization.diagnosis_config)

        optimization_info: dict = {
            "project_id": optimization.project_id,
            "id": optimization.id,
            "name": optimization.name,
            "precision": {
                "id": precision.id,
                "name": precision.name,
            },
            "optimization_type": {
                "id": optimization_type.id,
                "name": optimization_type.name,
            },
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
            },
            "config_path": optimization.config_path,
            "log_path": optimization.log_path,
            "execution_command": optimization.execution_command,
            "batch_size": optimization.batch_size,
            "sampling_size": optimization.sampling_size,
            "status": optimization.status,
            "created_at": str(optimization.created_at),
            "last_run_at": str(optimization.last_run_at),
            "duration": optimization.duration,
            "optimized_model": None,
            "accuracy_benchmark_id": optimization.accuracy_benchmark_id,
            "performance_benchmark_id": optimization.performance_benchmark_id,
            "tuning_details": None,
            "diagnosis_config": diagnosis_config,
        }
        if optimized_model is not None:
            model_info = Model.build_info(optimized_model)
            optimization_info.update(
                {
                    "optimized_model": model_info,
                },
            )

        if tuning_details is not None:
            acc_criterion_type = tuning_details.accuracy_criterion_type
            acc_criterion_threshold = tuning_details.accuracy_criterion_threshold
            optimization_data = {
                "tuning_details": {
                    "id": tuning_details.id,
                    "strategy": tuning_details.strategy,
                    "accuracy_criterion_type": acc_criterion_type,
                    "accuracy_criterion_threshold": acc_criterion_threshold,
                    "multi_objectives": tuning_details.objective,
                    "exit_policy": json.loads(tuning_details.exit_policy),
                    "random_seed": tuning_details.random_seed,
                    "created_at": str(tuning_details.created_at),
                    "modified_at": str(tuning_details.modified_at),
                    "tuning_history": None,
                },
            }
            if tuning_history is not None:
                optimization_data["tuning_details"].update(
                    {"tuning_history": TuningHistory.build_info(tuning_history)},
                )
            optimization_info.update(
                optimization_data,
            )
        return optimization_info

    @staticmethod
    def get_pinned_benchmarks(
        db_session: session.Session,
        optimization: Any,
    ) -> Dict[str, Optional[dict]]:
        """Get pinned benchmarks for optimization."""
        (results) = (
            db_session.query(
                Benchmark,
                BenchmarkResult,
                Model,
                Dataset,
            )
            .outerjoin(Benchmark.result)
            .join(Benchmark.model)
            .join(Benchmark.dataset)
            .filter(
                Benchmark.id.in_(
                    [
                        optimization.performance_benchmark_id,
                        optimization.accuracy_benchmark_id,
                    ],
                ),
            )
            .all()
        )
        accuracy_benchmark = None
        performance_benchmark = None
        for result in results:
            benchmark, benchmark_result, model, dataset = result
            benchmark_info = Benchmark.build_info(benchmark, benchmark_result, model, dataset)
            if benchmark.mode == Benchmarks.PERF:
                performance_benchmark = benchmark_info
            if benchmark.mode == Benchmarks.ACC:
                accuracy_benchmark = benchmark_info

        return {
            "accuracy": accuracy_benchmark,
            "performance": performance_benchmark,
        }


update_last_run_date_trigger = """
CREATE TRIGGER update_optimization_run_date AFTER UPDATE ON optimization
WHEN NEW.status LIKE "wip"
BEGIN
    UPDATE optimization SET last_run_at = datetime("now") WHERE id=NEW.id;
END;
"""

event.listen(Optimization.__table__, "after_create", DDL(update_last_run_date_trigger))
