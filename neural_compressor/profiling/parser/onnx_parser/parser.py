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
import json
from collections import defaultdict
from typing import List

from neural_compressor.profiling.parser.parser import ProfilingParser
from neural_compressor.profiling.parser.result import ProfilingResult


class OnnxProfilingParser(ProfilingParser):
    """Parser class is responsible for parsing profiling log files."""

    def process(self) -> List[dict]:
        """Process profiling logs.

        Returns:
            list of dicts with processed profiling results
        """
        summarized = defaultdict(
            lambda: {
                "node_name": "",
                "total_execution_time": 0,
                "accelerator_execution_time": 0,
                "cpu_execution_time": 0,
                "op_run": 0,
                "op_defined": 0,
            },
        )

        for log_file in self._logs:
            with open(log_file) as f:
                data = json.load(f)

            # Iterate over each node in the JSON data and update the summarized data
            for node in data:
                category = node.get("cat")  # Get the category of the node
                op_name = node.get("args", {}).get("op_name")  # Get the operation name of the node
                dur = int(node.get("dur") or 0)  # Get the duration of the node

                # Skip nodes with missing or invalid values
                if category is None or node.get("name") is None or op_name is None:
                    continue

                # Update the summarized data for this operation
                summarized[op_name]["node_name"] = op_name
                summarized[op_name]["total_execution_time"] += dur
                if category == "Node":
                    summarized[op_name]["cpu_execution_time"] += dur
                elif category == "kernel":
                    summarized[op_name]["accelerator_execution_time"] += dur
                summarized[op_name]["op_defined"] += 1
                summarized[op_name]["op_run"] += (
                    0 if not (dur or node.get("args", {}).get("thread_scheduling_stats")) else 1
                )

        for node in summarized.values():
            self.add_result(ProfilingResult(**node))

        return self._serialize_results()
