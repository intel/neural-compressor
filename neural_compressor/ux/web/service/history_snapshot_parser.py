# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
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
from typing import Optional


class HistorySnapshotParser:
    """History Snapshot Parser."""

    def __init__(self, history_snapshot: list, provide_performance: bool = False):
        """Create object."""
        self.history_snapshot = history_snapshot
        self.provide_performance = provide_performance

    def parse_history_snapshot(self) -> dict:
        """Parse provided history snapshot."""
        if not self.history_snapshot:
            return {}

        if 1 != len(self.history_snapshot):
            raise ValueError("Expected history snapshot with one entry only")

        first_item = self.history_snapshot[0]
        return {
            "baseline_accuracy": self.extract_accuracy(first_item.get("baseline")),
            "baseline_performance": self.extract_performance(first_item.get("baseline")),
            "last_tune_accuracy": self.extract_accuracy(first_item.get("last_tune_result")),
            "last_tune_performance": self.extract_performance(first_item.get("last_tune_result")),
            "best_tune_accuracy": self.extract_accuracy(first_item.get("best_tune_result")),
            "best_tune_performance": self.extract_performance(first_item.get("best_tune_result")),
            "history": [
                self.parse_history_item(history_item)
                for history_item in first_item.get("history", [])
            ],
        }

    def parse_history_item(self, history_item: dict) -> dict:
        """Parse single item from history."""
        return {
            "accuracy": self.extract_accuracy(history_item.get("tune_result")),
            "performance": self.extract_performance(history_item.get("tune_result")),
        }

    def extract_accuracy(self, measurements: Optional[tuple]) -> Optional[float]:
        """Extract accuracy metric from result."""
        if not measurements:
            return None
        return measurements[0]

    def extract_performance(self, measurements: Optional[tuple]) -> Optional[float]:
        """Extract performance metric from result if needed."""
        if not self.provide_performance:
            return None
        if not measurements:
            return None
        return measurements[1]
