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
"""Optimization class factory."""

from typing import Optional

from neural_compressor.ux.components.optimization import Optimizations
from neural_compressor.ux.components.optimization.graph_optimizer.graph_optimization import GraphOptimization
from neural_compressor.ux.components.optimization.optimization import Optimization
from neural_compressor.ux.components.optimization.tune.tuning import Tuning
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.workload.workload import Workload


class OptimizationFactory:
    """Optimization factory."""

    @staticmethod
    def get_optimization(workload: Workload, template_path: Optional[str] = None) -> Optimization:
        """Get optimization for specified workload."""
        optimization_map = {
            Optimizations.TUNING: Tuning,
            Optimizations.GRAPH: GraphOptimization,
        }
        optimization = optimization_map.get(workload.mode, None)
        if optimization is None:
            raise InternalException(f"Could not find optimization class for {workload.mode}")
        log.debug(f"Initializing {optimization.__name__} class.")
        return optimization(workload, template_path)
