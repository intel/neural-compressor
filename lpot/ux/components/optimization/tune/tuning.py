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
"""Tuning class."""

import os
from typing import List, Optional

from lpot.ux.components.optimization.optimization import Optimization
from lpot.ux.utils.workload.workload import Workload


class Tuning(Optimization):
    """Tuning class."""

    def __init__(
        self,
        workload: Workload,
        template_path: Optional[str] = None,
    ) -> None:
        """Initialize Tuning class."""
        super().__init__(workload, template_path)
        if template_path:
            self.command = ["python", template_path]

    def execute(self) -> None:
        """Execute tuning."""
        pass

    @property
    def optimization_script(self) -> str:
        """Get optimization script path."""
        return os.path.join(os.path.dirname(__file__), "tune_model.py")

    @property
    def parameters(self) -> List[str]:
        """Get optimization parameters."""
        return [
            "--input-graph",
            self.input_graph,
            "--output-graph",
            self.output_graph,
            "--config",
            self.config_path,
            "--framework",
            self.framework,
        ]
