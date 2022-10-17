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
# pylint: disable=no-member
"""The db_operation package contains interfaces used to control local database."""
from neural_compressor.ux.components.db_manager.db_operations.benchmark_api_interface import (
    BenchmarkAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.dataset_api_interface import (
    DatasetAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.diagnosis_api_interface import (
    DiagnosisAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.dictionaries_api_interface import (
    DictionariesAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.examples_api_interface import (
    ExamplesAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.model_api_interface import (
    ModelAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.optimization_api_interface import (
    OptimizationAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.profiling_api_interface import (
    ProfilingAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.project_api_interface import (
    ProjectAPIInterface,
)

_interfaces = [
    BenchmarkAPIInterface,
    DatasetAPIInterface,
    DiagnosisAPIInterface,
    DictionariesAPIInterface,
    ExamplesAPIInterface,
    ModelAPIInterface,
    OptimizationAPIInterface,
    ProfilingAPIInterface,
    ProjectAPIInterface,
]
