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
"""Edge class."""

from lpot.ux.utils.json_serializer import JsonSerializer

from .node import Node


class Edge(JsonSerializer):
    """Edge in graph."""

    def __init__(self, source: Node, target: Node) -> None:
        """Construct object."""
        super().__init__()
        self._source = source
        self._target = target

    @property
    def source(self) -> str:
        """Return source Node id."""
        return self._source.id

    @property
    def target(self) -> str:
        """Return target Node id."""
        return self._target.id
