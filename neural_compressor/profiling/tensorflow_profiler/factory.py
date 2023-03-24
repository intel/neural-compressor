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

from typing import Optional, Any

from neural_compressor.experimental.data.dataloaders.tensorflow_dataloader import \
    TensorflowDataLoader
from neural_compressor.model.tensorflow_model import TensorflowBaseModel
from neural_compressor.profiling.tensorflow_profiler.profiler import Profiler


class ProfilerFactory:
    """Profiler factory."""

    @staticmethod
    def get_profiler(
            model_path: Any,
            model: TensorflowBaseModel,
            dataloader: TensorflowDataLoader,
            log_file: Optional[str] = None,
            *args,
            **kwargs,
    ) -> Optional[Profiler]:
        """Get profiling for specified workload."""
        return Profiler(model_path, model, dataloader, log_file)
