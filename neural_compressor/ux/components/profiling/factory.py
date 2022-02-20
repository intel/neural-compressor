# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022 Intel Corporation
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


class ProfilerFactory:
    """Profiling factory."""

    @staticmethod
    def get_profiler(profiling_data: dict) -> Optional[Profiler]:
        """Get profiling for specified framework."""
        framework_profilers = {
            "tensorflow": TensorflowProfilerFactory.get_profiler,
        }

        framework = profiling_data.get("framework")

        if framework is None or framework_profilers.get(framework, None) is None:
            return None
        return framework_profilers[framework](profiling_data)
