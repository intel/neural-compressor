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
"""The TuningDetails class."""
import json
from typing import Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship, session
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.components.optimization.tune.tuning import (
    TuningDetails as TuningDetailsInterface,
)


class TuningDetails(Base):
    """INC Bench tuning details' table representation."""

    __tablename__ = "tuning_details"

    id = Column(Integer, primary_key=True, index=True, unique=True)
    strategy = Column(String(50))
    accuracy_criterion_type = Column(String(50), default="relative")
    accuracy_criterion_threshold = Column(Float, default=0.1)
    objective = Column(String(50))
    exit_policy = Column(String)
    random_seed = Column(Integer)
    tuning_history_id = Column(Integer, ForeignKey("tuning_history.id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    tuning_history = relationship(
        "TuningHistory",
        foreign_keys=[tuning_history_id],
        cascade="all, delete",
    )

    @staticmethod
    def add(
        db_session: session.Session,
        strategy: str,
        accuracy_criterion_type: str,
        accuracy_criterion_threshold: float,
        objective: str,
        exit_policy: dict,
        random_seed: int,
    ) -> int:
        """
        Add optimization to database.

        returns id of added optimization
        """
        new_tuning_details = TuningDetails(
            strategy=strategy,
            accuracy_criterion_type=accuracy_criterion_type,
            accuracy_criterion_threshold=accuracy_criterion_threshold,
            objective=objective,
            exit_policy=json.dumps(exit_policy),
            random_seed=random_seed,
        )
        db_session.add(new_tuning_details)
        db_session.flush()

        return int(new_tuning_details.id)

    @staticmethod
    def update(
        db_session: session.Session,
        tuning_details_id: int,
        tuning_details_data: TuningDetailsInterface,
    ) -> dict:
        """Update tuning details."""
        tuning_details = (
            db_session.query(TuningDetails).filter(TuningDetails.id == tuning_details_id).one()
        )
        tuning_details.strategy = tuning_details_data.strategy
        tuning_details.accuracy_criterion_type = tuning_details_data.accuracy_criterion.type
        tuning_details.accuracy_criterion_threshold = (
            tuning_details_data.accuracy_criterion.threshold
        )
        tuning_details.objective = tuning_details_data.objective
        tuning_details.exit_policy = json.dumps(tuning_details_data.exit_policy)
        tuning_details.random_seed = tuning_details_data.random_seed

        db_session.add(tuning_details)
        db_session.flush()

        return {
            "id": tuning_details.id,
            "strategy": tuning_details.strategy,
            "accuracy_criterion_type": tuning_details.accuracy_criterion_type,
            "accuracy_criterion_threshold": tuning_details.accuracy_criterion_threshold,
            "multi_objectives": tuning_details.objective,
            "exit_policy": json.loads(tuning_details.exit_policy),
            "random_seed": tuning_details.random_seed,
            "created_at": str(tuning_details.created_at),
            "modified_at": str(tuning_details.modified_at),
        }

    @staticmethod
    def delete_tuning_details(
        db_session: session.Session,
        tuning_details_id: int,
    ) -> Optional[int]:
        """Remove tuning_details from database."""
        tuning_details = (
            db_session.query(TuningDetails)
            .filter(TuningDetails.id == tuning_details_id)
            .one_or_none()
        )
        if tuning_details is None:
            return None
        db_session.delete(tuning_details)
        db_session.flush()

        return int(tuning_details.id)

    @staticmethod
    def update_tuning_history(
        db_session: session.Session,
        tuning_details_id: int,
        tuning_history_id: int,
    ) -> dict:
        """Update status of optimization."""
        tuning_details = (
            db_session.query(TuningDetails).filter(TuningDetails.id == tuning_details_id).one()
        )
        tuning_details.tuning_history_id = tuning_history_id
        db_session.add(tuning_details)
        db_session.flush()

        return {
            "id": tuning_details.id,
            "tuning_history_id": tuning_details.tuning_history_id,
        }
