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
import re
from typing import Any, List

from neural_compressor.profiling.parser.parser import ProfilingParser
from neural_compressor.profiling.parser.result import ProfilingResult

ROUND_PRECISION = 2


class TensorFlowProfilingParser(ProfilingParser):
    """Parser class is responsible for parsing profiling log files."""

    @staticmethod
    def unify_time(string_value: str) -> float:
        """Unify time with unit to micro seconds float value.

        Args:
            string_value: string containing time value with unit

        Returns:
            float with time value in micro seconds
        """
        search = re.search(r"(\d+(\.\d+)?)\s*(\w+)", string_value)

        if not search:
            raise Exception(f"Could not parse {string_value}")
        value = round(float(search.group(1)), ROUND_PRECISION)
        unit = search.group(3)

        unit_map = {"s": 10e5, "sec": 10e5, "ms": 10e2, "us": 1, "ns": 10e-4}

        unit_modifier = unit_map.get(unit, None)
        if unit_modifier is None:
            raise Exception(f'Unit "{unit}" not recognized.')

        return value * unit_modifier

    def process(self) -> List[dict]:
        """Process profiling logs.

        Returns:
            list of dicts with processed profiling results
        """
        profiling_header = self.patterns.get("profiling_header", None)
        profiling_entry = self.patterns.get("profiling_entry", None)
        if any(pattern is None for pattern in [profiling_header, profiling_entry]):
            raise Exception("Could not find patterns to search.")

        for log_file in self._logs:
            with open(log_file) as file:
                lines = file.readlines()
            for line in reversed(lines):
                header_search: Any = re.search(profiling_header, line)
                if header_search:
                    break
                entry_search: Any = re.search(profiling_entry, line)
                if entry_search:
                    node_name = str(entry_search.group(1))
                    total_execution_time = str(entry_search.group(2))
                    accelerator_execution_time = str(entry_search.group(8))
                    cpu_execution_time = str(entry_search.group(14))
                    op_run = int(entry_search.group(20))
                    op_defined = int(entry_search.group(21))
                    self.add_result(
                        ProfilingResult(
                            node_name=node_name,
                            total_execution_time=int(self.unify_time(total_execution_time)),
                            accelerator_execution_time=int(self.unify_time(accelerator_execution_time)),
                            cpu_execution_time=int(self.unify_time(cpu_execution_time)),
                            op_run=op_run,
                            op_defined=op_defined,
                        ),
                    )

        return self._serialize_results()

    @property
    def patterns(self) -> dict:
        """Set patterns to get metrics from lines.

        Returns:
            dict with regex patterns
        """
        return {
            "profiling_entry": r"^(\S+)\s+(\d+(\.\d+)?\w+)\s\((\d+(\.\d+)?)%,"
            r"\s(\d+(\.\d+)?)%\),\s+(\d+(\.\d+)?\w+)\s\((\d+(\.\d+)?)%,"
            r"\s(\d+(\.\d+)?)%\),\s+(\d+(\.\d+)?\w+)\s\((\d+(\.\d+)?)%,"
            r"\s(\d+(\.\d+)?)%\),\s+(\d+)\|(\d+)$",
            "profiling_header": r"node name \| total execution time \| accelerator execution time "
            r"\| cpu execution time \| op occurrence \(run\|defined\)",
        }
