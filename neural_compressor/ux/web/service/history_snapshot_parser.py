# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022 Intel Corporation
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

"""History Snapshot Parser Service."""
from typing import List, Optional

from neural_compressor.ux.components.db_manager.params_interfaces import (
    TuningHistoryInterface,
    TuningHistoryItemInterface,
)


class HistorySnapshotParser:
    """History Snapshot Parser."""

    def __init__(self, history_snapshot: list, provide_performance: bool = False):
        """Create object."""
        self.history_snapshot = history_snapshot
        self.provide_performance = provide_performance

    def parse_history_snapshot(self) -> TuningHistoryInterface:
        """Parse provided history snapshot."""
        tuning_history: TuningHistoryInterface = TuningHistoryInterface()
        if not self.history_snapshot:
            return tuning_history

        if 1 != len(self.history_snapshot):
            raise ValueError("Expected history snapshot with one entry only")

        first_item = self.history_snapshot[0]

        tuning_history.baseline_accuracy = self.extract_accuracy(
            first_item.get("baseline"),
        )  # type: ignore
        tuning_history.baseline_performance = self.extract_performance(
            first_item.get("baseline"),
        )  # type: ignore
        tuning_history.last_tune_accuracy = self.extract_accuracy(
            first_item.get("last_tune_result"),
        )  # type: ignore
        tuning_history.last_tune_performance = self.extract_performance(
            first_item.get("last_tune_result"),
        )  # type: ignore
        tuning_history.best_tune_accuracy = self.extract_accuracy(
            first_item.get("best_tune_result"),
        )  # type: ignore
        tuning_history.best_tune_performance = self.extract_performance(
            first_item.get("best_tune_result"),
        )  # type: ignore
        tuning_history.history = [
            self.parse_history_item(history_item) for history_item in first_item.get("history", [])
        ]
        return tuning_history

    def parse_history_item(self, history_item: dict) -> TuningHistoryItemInterface:
        """Parse single item from history."""
        tuning_history_item: TuningHistoryItemInterface = TuningHistoryItemInterface()
        tuning_history_item.accuracy = self.extract_accuracy(
            history_item.get("tune_result"),
        )  # type: ignore
        tuning_history_item.performance = self.extract_performance(
            history_item.get("tune_result"),
        )  # type: ignore

        return tuning_history_item

    def extract_accuracy(
        self,
        measurements: Optional[tuple],
    ) -> Optional[List[float]]:
        """Extract accuracy metric from result."""
        if not measurements:
            return None
        accuracy = measurements[0]
        if isinstance(accuracy, float):
            return [accuracy]
        if isinstance(accuracy, list):
            return accuracy
        return None

    def extract_performance(
        self,
        measurements: Optional[tuple],
    ) -> Optional[List[float]]:
        """Extract performance metric from result if needed."""
        if not self.provide_performance:
            return None
        if not measurements:
            return None
        performance = measurements[1]
        if isinstance(performance, float):
            return [performance]
        if isinstance(performance, list):
            return performance
        return None
