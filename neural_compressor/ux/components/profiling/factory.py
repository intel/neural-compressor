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
"""Profiling class factory."""
from typing import Optional

from neural_compressor.ux.components.profiling.profiler import Profiler
from neural_compressor.ux.components.profiling.tensorflow_profiler.factory import (
    ProfilerFactory as TensorflowProfilerFactory,
)
from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.utils.workload.workload import Workload


class ProfilerFactory:
    """Profiling factory."""

    @staticmethod
    def get_profiler(workload_id: str, model_path: str) -> Optional[Profiler]:
        """Get profiling for specified framework."""
        workload: Workload = Workdir(
            request_id=workload_id,
            overwrite=False,
        ).get_workload_object()
        framework_profilers = {
            "tensorflow": TensorflowProfilerFactory.get_profiler,
        }

        if workload.framework is None or framework_profilers.get(workload.framework, None) is None:
            return None
        return framework_profilers[workload.framework](workload_id, model_path)
