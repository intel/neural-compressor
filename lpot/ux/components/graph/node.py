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
"""Node class."""

from typing import Any, Dict, List

from lpot.ux.utils.json_serializer import JsonSerializer

from .attribute import Attribute


class Node(JsonSerializer):
    """Single Node in graph."""

    TYPE_SIMPLE_NODE = "node"

    def __init__(
        self,
        id: str,
        label: str,
        properties: Dict[str, Any] = {},
        attributes: List[Attribute] = [],
        groups: List[str] = [],
    ) -> None:
        """Construct object."""
        super().__init__()
        self.id = id
        self.label = label
        self.properties = properties
        self.attributes = attributes
        self.groups = groups
        self.node_type = self.TYPE_SIMPLE_NODE
        self._skip.append("groups")


class GroupNode(Node):
    """Node being a group of Nodes in Graph."""

    TYPE_GROUP_NODE = "group_node"

    def __init__(
        self,
        id: str,
        group_name: str,
    ) -> None:
        """Construct object."""
        super().__init__(
            id=id,
            label=group_name,
        )
        self.group = group_name
        self.node_type = self.TYPE_GROUP_NODE
