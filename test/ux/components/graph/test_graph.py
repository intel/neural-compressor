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
"""Test Graph classes."""

import unittest
import uuid

from lpot.ux.components.graph.graph import Graph
from lpot.ux.components.graph.node import Node


def _get_random_string() -> str:
    """Create random string to be used."""
    return uuid.uuid4().hex


def _get_random_node() -> Node:
    """Create a Node with random values."""
    return Node(id=_get_random_string(), label=_get_random_string())


class TestGraph(unittest.TestCase):
    """Test Graph class."""

    def test_empty_graph_has_no_nodes(self) -> None:
        """Test if empty graph has no nodes."""
        graph = Graph()
        self.assertEqual([], graph.nodes)

    def test_empty_graph_has_no_edges(self) -> None:
        """Test if empty graph has no edges."""
        graph = Graph()
        self.assertEqual([], graph.edges)

    def test_graph_with_one_node_returns_only_one_node(self) -> None:
        """Test if empty graph with one node returns it correctly."""
        node = _get_random_node()
        graph = Graph()
        graph.add_node(node)

        self.assertEqual([node], graph.nodes)

    def test_adding_edge_when_no_nodes_does_not_add_an_adge(self) -> None:
        """Test if adding an edge to empty graph fails as expected."""
        source_id = _get_random_string()
        target_id = _get_random_string()

        graph = Graph()
        result = graph.add_edge(source_id=source_id, target_id=target_id)

        self.assertEqual(False, result)
        self.assertEqual([], graph.edges)

    def test_adding_edge_between_nodes_works(self) -> None:
        """Test adding an edge."""
        source = _get_random_node()
        target = _get_random_node()

        graph = Graph()
        graph.add_node(source)
        graph.add_node(target)
        result = graph.add_edge(source_id=source.id, target_id=target.id)

        self.assertEqual(True, result)

        edges = graph.edges
        self.assertEqual(1, len(edges))

        edge = edges[0]
        self.assertEqual(source, edge._source)
        self.assertEqual(target, edge._target)

    def test_comparing_to_something_else_fails(self) -> None:
        """Test if comparing Graph to something other fails."""
        with self.assertRaises(NotImplementedError):
            Graph() == 1


if __name__ == "__main__":
    unittest.main()
