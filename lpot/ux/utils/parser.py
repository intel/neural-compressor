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

"""Parser for tuning."""
import re
from typing import Any, Dict

from lpot.ux.utils.logger import log
from lpot.ux.utils.templates.metric import Metric


class Parser:
    """Parser class is responsible for parsing log files."""

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
            "perf_latency_fp32": r"Latency:\s+(\d+(\.\d+)?)",
            "perf_latency_int8": r"Latency:\s+(\d+(\.\d+)?)",
            "perf_throughput_fp32": r"Throughput:\s+(\d+(\.\d+)?)",
            "perf_throughput_int8": r"Throughput:\s+(\d+(\.\d+)?)",
        }
