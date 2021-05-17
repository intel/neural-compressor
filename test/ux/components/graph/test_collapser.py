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
from typing import List

from lpot.ux.components.graph.collapser import Collapser
from lpot.ux.components.graph.graph import Graph
from lpot.ux.components.graph.node import GroupNode, Node


def _create_simple_node(name: str, groups: List[str]) -> Node:
    """Create Node."""
    return Node(
        id=name,
        label=name,
        groups=groups,
    )


def _create_group_node(group_name: str) -> GroupNode:
    """Create Node."""
    node_name = f"node_group_{group_name}"
    return GroupNode(
        id=node_name,
        group_name=group_name,
    )


class TestCollapser(unittest.TestCase):
    """Test Collapser class."""

    def test_collapsing_with_all_groups_expanded(self) -> None:
        """Test that no collapsing will happen when all groups are expanded."""
        source = self._get_source_graph()

        collapser = Collapser(["a", "a:b", "a:b:c"])
        collapsed = collapser.collapse(graph=source)

        self.assertEqual(source, collapsed)

    def test_collapsing_all_groups(self) -> None:
        """Test that no collapsing will happen when all groups are expanded."""
        source = self._get_source_graph()

        expected = Graph()
        expected.add_node(_create_simple_node("input", []))
        expected.add_node(_create_group_node("a"))
        expected.add_node(_create_simple_node("e", ["e"]))
        expected.add_node(_create_simple_node("output", []))
        expected.add_edge("input", "node_group_a")
        expected.add_edge("node_group_a", "output")
        expected.add_edge("input", "e")
        expected.add_edge("e", "output")

        collapser = Collapser([])
        collapsed = collapser.collapse(graph=source)

        self.assertEqual(expected, collapsed)

    def test_expanding_all_groups(self) -> None:
        """Test that no collapsing will happen when all groups are expanded."""
        source = self._get_source_graph()

        collapser = Collapser(["node_group_a", "node_group_a:b", "node_group_a:b:c"])
        collapsed = collapser.collapse(graph=source)

        self.assertEqual(source, collapsed)

    def test_collapsing_external_group(self) -> None:
        """Test that no collapsing will happen when all groups are expanded."""
        source = self._get_source_graph()

        expected = Graph()
        expected.add_node(_create_simple_node("input", []))
        expected.add_node(_create_simple_node("a:b", ["a"]))
        expected.add_node(_create_group_node("a:b"))
        expected.add_node(_create_simple_node("e", ["e"]))
        expected.add_node(_create_simple_node("output", []))
        expected.add_edge("input", "a:b")
        expected.add_edge("a:b", "node_group_a:b")
        expected.add_edge("node_group_a:b", "output")
        expected.add_edge("input", "e")
        expected.add_edge("e", "output")

        collapser = Collapser(["node_group_a"])
        collapsed = collapser.collapse(graph=source)

        self.assertEqual(expected, collapsed)

    def test_collapsing_internal_group(self) -> None:
        """Test that no collapsing will happen when all groups are expanded."""
        source = self._get_source_graph()

        expected = Graph()
        expected.add_node(_create_simple_node("input", []))
        expected.add_node(_create_group_node("a"))
        expected.add_node(_create_simple_node("e", ["e"]))
        expected.add_node(_create_simple_node("output", []))
        expected.add_edge("input", "node_group_a")
        expected.add_edge("node_group_a", "output")
        expected.add_edge("input", "e")
        expected.add_edge("e", "output")

        collapser = Collapser(["node_group_a:b"])
        collapsed = collapser.collapse(graph=source)

        self.assertEqual(expected, collapsed)

    def _get_source_graph(self) -> Graph:
        """Return a graph with a well-known contents."""
        graph = Graph()
        graph.add_node(_create_simple_node("input", []))
        graph.add_node(_create_simple_node("a:b", ["a"]))
        graph.add_node(_create_simple_node("a:b:c1", ["a", "a:b"]))
        graph.add_node(_create_simple_node("a:b:c2", ["a", "a:b"]))
        graph.add_node(_create_simple_node("a:b:c:d", ["a", "a:b", "a:b:c"]))
        graph.add_node(_create_simple_node("e", ["e"]))
        graph.add_node(_create_simple_node("output", []))

        graph.add_edge("input", "a:b")
        graph.add_edge("a:b", "a:b:c1")
        graph.add_edge("a:b", "a:b:c2")
        graph.add_edge("a:b:c1", "a:b:c:d")
        graph.add_edge("a:b:c2", "a:b:c:d")
        graph.add_edge("a:b:c:d", "output")
        graph.add_edge("input", "e")
        graph.add_edge("e", "output")

        return graph


if __name__ == "__main__":
    unittest.main()
