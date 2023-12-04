# -*- coding: utf-8 -*-
# Copyright (c) 2023 Intel Corporation
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

"""Parsers for log files."""
from abc import ABC, abstractmethod
from typing import List

from neural_compressor.profiling.parser.result import ProfilingResult


class ProfilingParser(ABC):
    """Parser class is responsible for parsing profiling log files."""

    def __init__(self, logs: list) -> None:
        """Initialize parser.

        Args:
            logs: list of path to logs

        Returns:
            None
        """
        self._logs = logs
        self._results: List[ProfilingResult] = []

    @property
    def results(self) -> List[ProfilingResult]:
        """Get profiling results.

        Returns: list of ProfilingResult entries
        """
        return self._results

    def add_result(self, result: ProfilingResult) -> None:
        """Add result to list.

        Args:
            result: ProfilingResult object

        Returns:
            None
        """
        self._results.append(result)

    @abstractmethod
    def process(self) -> List[dict]:
        """Process profiling logs.

        Returns:
            list of dicts with processed profiling results
        """
        raise NotImplementedError

    def _serialize_results(self) -> list:
        """Serialize Results list.

        Returns:
            list of serialized results
        """
        sorted_results = sorted(
            self.results,
            key=lambda result: result.total_execution_time,
            reverse=True,
        )
        return [result.serialize() for result in sorted_results]
