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
"""The TuningHistory class."""
import json
from typing import Any, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import session
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base


class TuningHistory(Base):
    """INC Bench tuning history' table representation."""

    __tablename__ = "tuning_history"

    id = Column(Integer, primary_key=True, index=True, unique=True)
    minimal_accuracy = Column(Float)

    baseline_accuracy = Column(String)
    baseline_performance = Column(String)

    last_tune_accuracy = Column(String)
    last_tune_performance = Column(String)

    best_tune_accuracy = Column(String)
    best_tune_performance = Column(String)

    history = Column(String)

    created_at = Column(DateTime, nullable=False, default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    @staticmethod
    def add(
        db_session: session.Session,
        minimal_accuracy: Optional[float],
        baseline_accuracy: Optional[List[float]],
        baseline_performance: Optional[List[float]],
        last_tune_accuracy: Optional[List[float]],
        last_tune_performance: Optional[List[float]],
        best_tune_accuracy: Optional[List[float]],
        best_tune_performance: Optional[List[float]],
        history: List[dict],
    ) -> int:
        """
        Add tuning history to database.

        returns id of added tuning history
        """
        new_tuning_history = TuningHistory(
            minimal_accuracy=minimal_accuracy,
            baseline_accuracy=json.dumps(baseline_accuracy),
            baseline_performance=json.dumps(baseline_performance),
            last_tune_accuracy=json.dumps(last_tune_accuracy),
            last_tune_performance=json.dumps(last_tune_performance),
            best_tune_accuracy=json.dumps(best_tune_accuracy),
            best_tune_performance=json.dumps(best_tune_performance),
            history=json.dumps(history),
        )
        db_session.add(new_tuning_history)
        db_session.flush()

        return int(new_tuning_history.id)

    @staticmethod
    def build_info(tuning_history: Any) -> dict:
        """Build tuning history info."""
        tuning_history = {
            "id": tuning_history.id,
            "minimal_accuracy": tuning_history.minimal_accuracy,
            "baseline_accuracy": json.loads(tuning_history.baseline_accuracy),
            "baseline_performance": json.loads(tuning_history.baseline_performance),
            "last_tune_accuracy": json.loads(tuning_history.last_tune_accuracy),
            "last_tune_performance": json.loads(tuning_history.last_tune_performance),
            "best_tune_accuracy": json.loads(tuning_history.best_tune_accuracy),
            "best_tune_performance": json.loads(tuning_history.best_tune_performance),
            "history": json.loads(tuning_history.history),
        }

        return tuning_history
