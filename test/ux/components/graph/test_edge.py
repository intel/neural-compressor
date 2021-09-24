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
"""Test Edge."""

import unittest
import uuid

from neural_compressor.ux.components.graph.edge import Edge
from neural_compressor.ux.components.graph.node import Node
from neural_compressor.ux.utils.json_serializer import JsonSerializer


def _get_random_string() -> str:
    """Create random string to be used."""
    return uuid.uuid4().hex


def _get_random_node() -> Node:
    """Create a Node with random values."""
    return Node(id=_get_random_string(), label=_get_random_string())


class TestEdge(unittest.TestCase):
    """Test Edge class."""

    def test_setting_parameters(self) -> None:
        """Test if parameters are correctly set."""
        source_node = _get_random_node()
        target_node = _get_random_node()

        edge = Edge(
            source=source_node,
            target=target_node,
        )

        self.assertEqual(source_node, edge._source)
        self.assertEqual(source_node.id, edge.source)
        self.assertEqual(target_node, edge._target)
        self.assertEqual(target_node.id, edge.target)

    def test_serialization(self) -> None:
        """Test if Edge is serialized as expected."""
        source_node = _get_random_node()
        target_node = _get_random_node()

        edge = Edge(
            source=source_node,
            target=target_node,
        )

        expected_serialized_object = {
            "source": source_node.id,
            "target": target_node.id,
        }
        serialized = JsonSerializer.serialize_item(edge)
        self.assertEqual(expected_serialized_object, serialized)


if __name__ == "__main__":
    unittest.main()
