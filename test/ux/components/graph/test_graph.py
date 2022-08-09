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
from typing import List

from neural_compressor.ux.components.graph.graph import Graph
from neural_compressor.ux.components.graph.node import Node


def _get_random_string() -> str:
    """Create random string to be used."""
    return uuid.uuid4().hex


def _get_random_node() -> Node:
    """Create a Node with random values."""
    return Node(id=_get_random_string(), label=_get_random_string())


def _get_fake_graph() -> Graph:
    """Create graph with fake data."""
    node_1 = Node(id="node_1", label="Input")
    node_2 = Node(id="node_2", label="SomeNode")
    node_3 = Node(id="node_3", label="SomeNode")
    node_4 = Node(id="node_4", label="SomeNode")
    node_5 = Node(id="node_5", label="Softmax")

    graph = Graph()
    graph.add_node(node_1)
    graph.add_node(node_2)
    graph.add_node(node_3)
    graph.add_node(node_4)
    graph.add_node(node_5)

    graph.add_edge(source_id=node_1.id, target_id=node_2.id)
    graph.add_edge(source_id=node_1.id, target_id=node_3.id)
    graph.add_edge(source_id=node_1.id, target_id=node_4.id)
    graph.add_edge(source_id=node_2.id, target_id=node_5.id)
    graph.add_edge(source_id=node_3.id, target_id=node_5.id)
    graph.add_edge(source_id=node_4.id, target_id=node_5.id)

    return graph


def _get_fake_sequential_graph() -> Graph:
    """Create graph with fake data."""
    node_1 = Node(id="node_1", label="Input")
    node_2 = Node(id="node_2", label="Conv2D")
    node_3 = Node(id="node_3", label="AddBias")
    node_4 = Node(id="node_4", label="ReLU")
    node_5 = Node(id="node_5", label="Softmax")

    graph = Graph()
    graph.add_node(node_1)
    graph.add_node(node_2)
    graph.add_node(node_3)
    graph.add_node(node_4)
    graph.add_node(node_5)

    graph.add_edge(source_id=node_1.id, target_id=node_2.id)
    graph.add_edge(source_id=node_2.id, target_id=node_3.id)
    graph.add_edge(source_id=node_3.id, target_id=node_4.id)
    graph.add_edge(source_id=node_4.id, target_id=node_5.id)

    return graph


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

    def test_highlight_pattern(self) -> None:
        """Test highlight_pattern."""
        graph = _get_fake_sequential_graph()
        op_name = "node_2"
        pattern = ["Conv2D", "AddBias", "ReLU"]

        graph.highlight_pattern(op_name, pattern)

        self.assertFalse(graph.get_node("node_1").highlight)
        self.assertTrue(graph.get_node("node_2").highlight)
        self.assertTrue(graph.get_node("node_3").highlight)
        self.assertTrue(graph.get_node("node_4").highlight)
        self.assertFalse(graph.get_node("node_5").highlight)

    def test_get_multiple_target_nodes(self) -> None:
        """Test getting multiple target nodes using get_target_nodes."""
        graph = _get_fake_graph()
        result = [node.id for node in graph.get_target_nodes("node_1")]

        expected = ["node_2", "node_3", "node_4"]

        self.assertListEqual(result, expected)

    def test_get_single_target_node(self) -> None:
        """Test getting single target node using get_target_nodes."""
        graph = _get_fake_graph()
        result = [node.id for node in graph.get_target_nodes("node_3")]

        expected = ["node_5"]

        self.assertListEqual(result, expected)

    def test_get_target_node_for_last_node(self) -> None:
        """Test getting target node for last node using get_target_nodes."""
        graph = _get_fake_graph()
        result = [node.id for node in graph.get_target_nodes("node_5")]

        expected: List[str] = []

        self.assertListEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
