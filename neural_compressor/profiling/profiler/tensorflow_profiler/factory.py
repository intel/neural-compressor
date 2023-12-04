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

from neural_compressor.data.dataloaders.tensorflow_dataloader import TensorflowDataLoader
from neural_compressor.model.tensorflow_model import TensorflowBaseModel
from neural_compressor.profiling.profiler.profiler import Profiler
from neural_compressor.profiling.profiler.tensorflow_profiler.profiler import Profiler as FrozenPbProfiler


class ProfilerFactory:
    """Profiler factory."""

    @staticmethod
    def get_profiler(
        model: TensorflowBaseModel,
        dataloader: TensorflowDataLoader,
        log_file: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Profiler:
        """Get profiling for specified framework.

        Args:
            model: model to be profiled
            dataloader: DataLoader object
            log_file: optional path to log file

        Returns:
            Profiler instance if model is supported else None
        """
        framework_profilers = {
            "frozen_pb": FrozenPbProfiler,
        }

        profiler = framework_profilers.get(model.model_type, None)
        if profiler is None:
            raise Exception(f"Profiling for '{model.model_type}' model type is not supported.")
        return profiler(model, dataloader, log_file)
