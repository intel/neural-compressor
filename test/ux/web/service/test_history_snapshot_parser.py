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
"""HistorySnapshotParser test."""

import unittest

from neural_compressor.ux.web.service.history_snapshot_parser import HistorySnapshotParser


class TestWorkloadService(unittest.TestCase):
    """Test HistorySnapshotParser."""

    HISTORY_SNAPSHOT = {
        "baseline": (0.12, [0.34]),
        "last_tune_result": (0.56, [0.78]),
        "best_tune_result": (0.90, [0.10]),
        "history": [
            {
                "tune_result": (0.11, [0.12]),
            },
            {
                "there_is_no_tune_result_here": (0.13, [0.14]),
            },
            {
                "tune_result": (0.15, [0.16]),
            },
        ],
    }

    def test_parsing_empty_history(self) -> None:
        """Test parsing history."""
        parser = HistorySnapshotParser([])
        result = parser.parse_history_snapshot().serialize()

        expected: dict = {
            "minimal_accuracy": None,
            "baseline_accuracy": None,
            "baseline_performance": None,
            "last_tune_accuracy": None,
            "last_tune_performance": None,
            "best_tune_accuracy": None,
            "best_tune_performance": None,
            "history": [],
        }

        self.assertDictEqual(expected, result)

    def test_parsing_history_with_many_entries_fails(self) -> None:
        """Test parsing history."""
        parser = HistorySnapshotParser([1, 2])
        with self.assertRaisesRegex(ValueError, "Expected history snapshot with one entry only"):
            parser.parse_history_snapshot()

    def test_parsing_history_without_values(self) -> None:
        """Test parsing history."""
        parser = HistorySnapshotParser([{}])
        expected: dict = {
            "minimal_accuracy": None,
            "baseline_accuracy": None,
            "baseline_performance": None,
            "last_tune_accuracy": None,
            "last_tune_performance": None,
            "best_tune_accuracy": None,
            "best_tune_performance": None,
            "history": [],
        }
        result = parser.parse_history_snapshot().serialize()
        self.assertIs(type(result), dict)
        self.assertDictEqual(expected, result)

    def test_parsing_history_with_performance(self) -> None:
        """Test parsing history."""
        parser = HistorySnapshotParser([self.HISTORY_SNAPSHOT], True)
        expected = {
            "minimal_accuracy": None,
            "baseline_accuracy": [0.12],
            "baseline_performance": [0.34],
            "last_tune_accuracy": [0.56],
            "last_tune_performance": [0.78],
            "best_tune_accuracy": [0.90],
            "best_tune_performance": [0.10],
            "history": [
                {
                    "accuracy": [0.11],
                    "performance": [0.12],
                },
                {
                    "accuracy": None,
                    "performance": None,
                },
                {
                    "accuracy": [0.15],
                    "performance": [0.16],
                },
            ],
        }

        result = parser.parse_history_snapshot().serialize()
        self.assertIs(type(result), dict)
        self.assertDictEqual(expected, result)

    def test_parsing_history_without_performance(self) -> None:
        """Test parsing history."""
        parser = HistorySnapshotParser([self.HISTORY_SNAPSHOT], False)
        expected = {
            "minimal_accuracy": None,
            "baseline_accuracy": [0.12],
            "baseline_performance": None,
            "last_tune_accuracy": [0.56],
            "last_tune_performance": None,
            "best_tune_accuracy": [0.90],
            "best_tune_performance": None,
            "history": [
                {
                    "accuracy": [0.11],
                    "performance": None,
                },
                {
                    "accuracy": None,
                    "performance": None,
                },
                {
                    "accuracy": [0.15],
                    "performance": None,
                },
            ],
        }

        result = parser.parse_history_snapshot().serialize()
        self.assertIs(type(result), dict)
        self.assertDictEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
