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
from typing import Mapping, Type

from neural_compressor.ux.components.optimization.graph_optimizer.graph_optimization import (
    GraphOptimization,
)
from neural_compressor.ux.components.optimization.mixed_precision.mixed_precision import (
    MixedPrecision,
)
from neural_compressor.ux.components.optimization.optimization import Optimization
from neural_compressor.ux.components.optimization.pruning.pruning import Pruning
from neural_compressor.ux.components.optimization.tune.tuning import Tuning
from neural_compressor.ux.utils.consts import OptimizationTypes
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.logger import log


class OptimizationFactory:
    """Optimization factory."""

    @staticmethod
    def get_optimization(
        optimization_data: dict,
        project_data: dict,
        dataset_data: dict,
    ) -> Optimization:
        """Get optimization for specified workload."""
        try:
            optimization_type = optimization_data["optimization_type"]["name"]
        except KeyError:
            raise InternalException("Missing optimization type.")
        optimization_map: Mapping[str, Type[Optimization]] = {
            OptimizationTypes.QUANTIZATION.value: Tuning,
            OptimizationTypes.GRAPH_OPTIMIZATION.value: GraphOptimization,
            OptimizationTypes.MIXED_PRECISION.value: MixedPrecision,
            OptimizationTypes.PRUNING.value: Pruning,
        }
        optimization = optimization_map.get(optimization_type, None)
        if optimization is None:
            raise InternalException(f"Could not find optimization class for {optimization_type}")
        log.debug(f"Initializing {optimization.__name__} class.")
        return optimization(optimization_data, project_data, dataset_data)
