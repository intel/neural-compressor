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

"""Profiling Result class."""


class ProfilingResult:
    """Profiling result class."""

    def __init__(
            self,
            node_name: str,
            total_execution_time: int,
            accelerator_execution_time: int,
            cpu_execution_time: int,
            op_run: int,
            op_defined: int,
    ) -> None:
        """Create profiling result instance."""
        self.node_name = node_name
        self.total_execution_time = total_execution_time
        self.accelerator_execution_time = accelerator_execution_time
        self.cpu_execution_time = cpu_execution_time
        self.op_run = op_run
        self.op_defined = op_defined

    def serialize(self) -> dict:
        """Serialize profiling result."""
        return {
            "node_name": self.node_name,
            "total_execution_time": self.total_execution_time,
            "accelerator_execution_time": self.accelerator_execution_time,
            "cpu_execution_time": self.cpu_execution_time,
            "op_run": self.op_run,
            "op_defined": self.op_defined,
        }
