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
"""Graph reader."""


from typing import List

from neural_compressor.ux.components.graph.collapser import Collapser
from neural_compressor.ux.components.graph.graph import Graph
from neural_compressor.ux.components.model.repository import ModelRepository


class GraphReader:
    """Model graph reader."""

    def read(self, model_path: str, expanded_groups: List[str]) -> Graph:
        """Return Graph for given model path."""
        model_repository = ModelRepository()
        model = model_repository.get_model(model_path)
        graph = model.get_model_graph()

        collapser = Collapser(expanded_groups)
        collapsed_graph = collapser.collapse(graph)

        return collapsed_graph
