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

from .collapser import Collapser
from .graph import Graph
from .reader.repository import GraphReaderRepository


class GraphReader:
    """Model graph reader."""

    def __init__(self) -> None:
        """Construct object."""
        self.graph_reader_repository = GraphReaderRepository()

    def ensure_model_readable(self, model_path: str) -> None:
        """Decide if provided model can be read to graph."""
        reader = self.graph_reader_repository.find(model_path)
        reader.ensure_model_readable(model_path)

    def read(self, model_path: str, expanded_groups: List[str]) -> Graph:
        """Return Graph for given model path."""
        reader = self.graph_reader_repository.find(model_path)
        graph = reader.read(model_path)

        collapser = Collapser(expanded_groups)
        collapsed_graph = collapser.collapse(graph)

        return collapsed_graph
