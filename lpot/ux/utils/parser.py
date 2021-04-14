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

"""Parsers for log files."""
import re
from typing import Any, Dict, List, Union

from lpot.ux.utils.logger import log
from lpot.ux.utils.templates.metric import Metric


class TuningParser:
    """Parser class is responsible for parsing tuning log files."""

    def __init__(self, logs: list) -> None:
        """Initialize object."""
        self._logs = logs
        self.metric = Metric()

    def process(self) -> Dict[str, Any]:
        """Process files."""
        for log_file in self._logs:
            log.debug(f"Read from {log_file}")

            with open(log_file) as f:
                for line in f:
                    for key in self.patterns:
                        prog = re.compile(self.patterns[key])
                        match = prog.search(line)
                        if match:
                            self.metric.insert_data(key, match.group(1))
        parsed_data: Dict[str, Any] = self.metric.serialize()  # type: ignore
        return parsed_data

    @property
    def patterns(self) -> dict:
        """Set patterns to get metrics from lines."""
        return {
            "acc_fp32": r".*FP32 baseline is: \[(\d+.\d+),",
            "acc_int8": r".*Best tune result is: \[(\d+.\d+),",
        }


class BenchmarkParser:
    """Parser class is responsible for parsing benchmark log files."""

    def __init__(self, logs: list) -> None:
        """Initialize object."""
        self._logs = logs

    def process(self) -> Dict[str, Any]:
        """Process files."""
        metric = Metric()
        partial: Dict[str, List] = {}
        for log_file in self._logs:
            log.debug(f"Read from {log_file}")

            with open(log_file) as f:
                for line in f:
                    for key in self.patterns:
                        prog = re.compile(self.patterns[key])
                        match = prog.search(line)
                        if not match:
                            continue
                        metric_name = f"perf_{key}_fp32"
                        metric.insert_data(metric_name, match.group(1))
                        converted_value = getattr(metric, metric_name)
                        parse_result = {
                            key: converted_value,
                        }
                        partial = self.update_partial(partial, parse_result)

        return self.summarize_partial(partial)

    def update_partial(
        self,
        partial: Dict[str, List],
        parsed_result: Dict[str, Union[float, int]],
    ) -> Dict[str, List]:
        """Update partial entries."""
        for key, value in parsed_result.items():
            if key not in partial:
                partial[key] = []
            partial[key].append(value)
        return partial

    def summarize_partial(self, partial: dict) -> dict:
        """Calculate final values."""
        summary = {}
        for key, value in partial.items():
            summarized_value = self.summarize_value(key, value)
            for precision in ["fp32", "int8"]:
                metric_name = f"perf_{key}_{precision}"

                summary[metric_name] = summarized_value
        return summary

    def summarize_value(self, key: str, value: list) -> Union[float, int]:
        """Calculate final value."""
        if key == "latency":
            return round(sum(value) / len(value), 4)
        if key == "throughput":
            return round(sum(value), 4)
        return value[0]

    @property
    def patterns(self) -> dict:
        """Set patterns to get metrics from lines."""
        return {
            "throughput": r"Throughput:\s+(\d+(\.\d+)?)",
            "latency": r"Latency:\s+(\d+(\.\d+)?)",
        }
