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
from abc import ABC
from typing import Any, Dict, List, Union

from neural_compressor.ux.components.benchmark import Benchmarks
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.templates.metric import Metric


class Parser(ABC):
    """Parser abstract class."""

    def __init__(self, logs: list) -> None:
        """Initialize parser."""
        self._logs = logs
        self.metric = Metric()

    def process(self) -> Dict[str, Any]:
        """Process log files."""
        raise NotImplementedError

    @property
    def patterns(self) -> dict:
        """Set patterns to get metrics from lines."""
        raise NotImplementedError


class OptimizationParser(Parser):
    """Parser class is responsible for parsing optimization log files."""

    def process(self) -> Dict[str, Any]:
        """Process files."""
        for log_file in self._logs:
            log.debug(f"Read from {log_file}")

            with open(log_file) as f:
                for line in f:
                    for key in self.patterns:
                        prog = re.compile(self.patterns[key])
                        match = prog.search(line)
                        if match and match.groupdict().get(key):
                            requested_value = str(match.groupdict().get(key))
                            self.metric.insert_data(key, requested_value)
        parsed_data: Dict[str, Any] = self.metric.serialize()  # type: ignore
        return parsed_data

    @property
    def patterns(self) -> dict:
        """Set patterns to get metrics from lines."""
        return {
            "acc_input_model": r".*FP32 baseline is:\s+\[("
            r"(accuracy:\s+(?P<acc_input_model>(\d+(\.\d+)?)))?"
            r"(duration\s+\(seconds\):\s+(?P<duration>(\d+(\.\d+)?)))?"
            r"(memory footprint\s+\(MB\):\s+(?P<mem_footprint>(\d+(\.\d+)?)))?(,\s+)?"
            r")*\]",
            "acc_optimized_model": r".*Best tune result is:\s+\[("
            r"(accuracy:\s+(?P<acc_optimized_model>(\d+(\.\d+)?)))?"
            r"(duration\s+\(seconds\):\s+(?P<duration>(\d+(\.\d+)?)))?"
            r"(memory footprint\s+\(MB\):\s+(?P<mem_footprint>(\d+(\.\d+)?)))?(,\s+)?"
            r")*\]",
            "path_optimized_model": r".*Save quantized model to (?P<path_optimized_model>.*)\.",
        }


class PerformanceParser(Parser):
    """Parser class is responsible for parsing performance benchmark log files."""

    def process(self) -> Dict[str, Any]:
        """Process files."""
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
                        metric_name = f"perf_{key}_input_model"
                        self.metric.insert_data(metric_name, match.group(1))
                        converted_value = getattr(self.metric, metric_name)
                        parse_result = {
                            key: converted_value,
                        }
                        partial = self.update_partial(partial, parse_result)

        return self.summarize_partial(partial)

    @staticmethod
    def update_partial(
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
            for precision in ["input_model", "optimized_model"]:
                metric_name = f"perf_{key}_{precision}"

                summary[metric_name] = summarized_value
        return summary

    @staticmethod
    def summarize_value(key: str, value: list) -> Union[float, int]:
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


class AccuracyParser(Parser):
    """Parser class is responsible for parsing accuracy benchmark log files."""

    def process(self) -> Dict[str, Any]:
        """Process accuracy logs."""
        for log_file in self._logs:
            log.debug(f"Read from {log_file}")

            with open(log_file) as f:
                for line in f:
                    for key in self.patterns:
                        prog = re.compile(self.patterns[key])
                        match = prog.search(line)
                        if match:
                            for precision in ["input_model", "optimized_model"]:
                                metric_name = f"acc_{precision}"
                                self.metric.insert_data(metric_name, match.group(1))

        parsed_data: Dict[str, Any] = self.metric.serialize()  # type: ignore
        return parsed_data

    @property
    def patterns(self) -> dict:
        """Set patterns to get metrics from lines."""
        return {
            Benchmarks.ACC: r"Accuracy is (\d+(\.\d+)?)",
        }


class BenchmarkParserFactory:
    """Benchmark parser factory."""

    @staticmethod
    def get_parser(benchmark_mode: str, logs: List[str]) -> Parser:
        """Get benchmark parser for specified mode."""
        parser_map = {
            Benchmarks.PERF: PerformanceParser,
            Benchmarks.ACC: AccuracyParser,
        }
        parser = parser_map.get(benchmark_mode, None)
        if parser is None:
            raise InternalException(f"Could not find optimization class for {benchmark_mode}")
        return parser(logs)
