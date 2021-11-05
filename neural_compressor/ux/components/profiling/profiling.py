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
"""Profiling class."""

import os
import re
from typing import Any, Dict, Optional

from neural_compressor.ux.components.profiling.factory import ProfilerFactory
from neural_compressor.ux.components.profiling.profiler import Profiler
from neural_compressor.ux.utils.workload.workload import Workload


class Profiling:
    """Profiling class."""

    def __init__(
        self,
        workload: Workload,
        model_path: str,
    ) -> None:
        """Initialize Benchmark class."""
        self.workload = workload
        self.model_path = model_path
        self.profiling_script = os.path.join(
            os.path.dirname(__file__),
            "profile_model.py",
        )

        self.command = [
            "python",
            self.profiling_script,
            "--workload-id",
            self.workload.id,
            "--model-path",
            self.model_path,
        ]

    def execute(self) -> None:
        """Execute benchmark and collect results."""
        profiler: Optional[Profiler] = ProfilerFactory.get_profiler(
            workload_id=self.workload.id,
            model_path=self.model_path,
        )
        if profiler is None:
            raise Exception("Could not find profiler for workload.")
        profiler.profile_model()

    def serialize(self) -> Dict[str, Any]:
        """Serialize Profiling class to dict."""
        result = {}
        for key, value in self.__dict__.items():
            variable_name = re.sub(r"^_", "", key)
            if variable_name == "command":
                result[variable_name] = " ".join(value)
            else:
                result[variable_name] = value
        return result
