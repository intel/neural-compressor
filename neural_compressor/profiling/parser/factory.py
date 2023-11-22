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
"""Profiling parser class factory."""
from typing import Optional

from neural_compressor.model import BaseModel
from neural_compressor.profiling.parser.onnx_parser.factory import OnnxrtParserFactory
from neural_compressor.profiling.parser.parser import ProfilingParser
from neural_compressor.profiling.parser.tensorflow_parser.factory import TensorFlowParserFactory


class ParserFactory:
    """Parser factory."""

    @staticmethod
    def get_parser(
        model: BaseModel,
        logs: list,
    ) -> Optional[ProfilingParser]:
        """Get parser for specified framework.

        Args:
            model: model to be profiled
            logs: list of path to logs

        Returns:
            ProfilingParser instance if model is supported else None
        """
        framework_parser = {
            "tensorflow": TensorFlowParserFactory.get_parser,
            "onnxruntime": OnnxrtParserFactory.get_parser,
        }

        parser = framework_parser.get(model.framework(), None)
        if parser is None:
            raise Exception(f"Profiling Parser for '{model.framework()}' framework is not supported.")
        return parser(logs)
