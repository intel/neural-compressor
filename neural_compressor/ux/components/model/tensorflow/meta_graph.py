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
"""Tensorflow frozen pb model."""

from typing import Any, List, Optional

from neural_compressor.ux.components.graph.graph import Graph
from neural_compressor.ux.components.model.model_type_getter import get_model_type
from neural_compressor.ux.components.model.tensorflow.model import TensorflowModel as TFModel


class MetaGraphModel(TFModel):
    """MetaGraph model."""

    def get_input_nodes(self) -> Optional[List[Any]]:
        """Get model input nodes."""
        return []

    def get_output_nodes(self) -> Optional[List[Any]]:
        """Get model output nodes."""
        return []

    def get_model_graph(self) -> Graph:
        """Get model graph."""
        raise NotImplementedError("Reading graph for MetaGraph is not supported.")

    @staticmethod
    def supports_path(path: str) -> bool:
        """Check if given path is of supported model."""
        return "checkpoint" == get_model_type(path)
