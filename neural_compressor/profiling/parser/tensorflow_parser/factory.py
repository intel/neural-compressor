# -*- coding: utf-8 -*-
# Copyright (c) 2023 Intel Corporation
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
"""Parser class factory."""

from typing import List, Optional

from neural_compressor.profiling.parser.tensorflow_parser.parser import TensorFlowProfilingParser


class TensorFlowParserFactory:
    """Parser factory."""

    @staticmethod
    def get_parser(
        logs: List,
        *args,
        **kwargs,
    ) -> Optional[TensorFlowProfilingParser]:
        """Get ProfilingParser for specified workload.

        Args:
            logs: list of path to logs
            *args: list with additional arguments
            **kwargs: dict with named arguments

        Returns:
            TensorFlowProfilingParser instance if model is supported else None
        """
        return TensorFlowProfilingParser(logs)
