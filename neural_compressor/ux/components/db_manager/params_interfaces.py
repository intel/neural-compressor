# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
"""Parameters interfaces for DB."""
from typing import List, Optional, Union

from neural_compressor.ux.components.optimization.tune.tuning import (
    TuningDetails as TuningDetailsInterface,
)


class ModelAddParamsInterface:
    """Interface for parameters to add model."""

    project_id: int
    name: str
    path: str
    framework: str
    size: float
    precision_id: int
    domain_id: int
    domain_flavour_id: int
    input_nodes: List[str]
    output_nodes: List[str]
    supports_profiling: bool
    supports_graph: bool

    @staticmethod
    def parse_nodes(nodes: Union[None, str, List[str]]) -> List[str]:
        """Parse model's boundary nodes."""
        if isinstance(nodes, str):
            return nodes.split(",")
        if isinstance(nodes, list):
            return nodes
        return []


class DatasetAddParamsInterface:
    """Interface for parameters to add dataset."""

    project_id: int
    dataset_name: str
    dataset_type: str
    parameters: dict
    transforms: dict
    filter: Optional[dict]
    metric: dict


class OptimizationAddParamsInterface:
    """Interface for parameters to add optimization."""

    project_id: int
    name: str
    precision_id: int
    optimization_type_id: int
    dataset_id: int
    batch_size: int
    sampling_size: int
    tuning_details: TuningDetailsInterface


class BenchmarkAddParamsInterface:
    """Interface for parameters to add benchmark."""

    name: str
    project_id: int
    model_id: int
    dataset_id: int
    mode: str
    batch_size: int
    iterations: int
    warmup_iterations: int
    number_of_instance: int
    cores_per_instance: int


class ProfilingAddParamsInterface:
    """Interface for parameters to add profiling."""

    name: str
    project_id: int
    model_id: int
    dataset_id: int
    num_threads: int


class ProfilingResultAddParamsInterface:
    """Interface for parameters to add profiling result."""

    profiling_id: int
    node_name: str
    total_execution_time: int
    accelerator_execution_time: int
    cpu_execution_time: int
    op_run: int
    op_defined: int
