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
"""Graph collapser."""

from typing import Dict, List, Set

from .graph import Graph
from .node import GroupNode, Node


class Collapser:
    """Graph collapser."""

    GROUP_NAME_PREFIX = "node_group_"

    def __init__(self, expanded_groups: List[str] = []) -> None:
        """Construct the collapser."""
        restored_group_names = [self._unprepare_group_name(name) for name in expanded_groups]
        self.expanded_groups: Set[str] = set(restored_group_names)
        self.group_sizes: Dict[str, int] = {}

    def collapse(self, graph: Graph) -> Graph:
        """Return graph with collapsed nodes."""
        collapsed_graph = Graph()

        self._calculate_group_sizes(graph)
        self._add_nodes_to_collapsed_graph(collapsed_graph, graph)
        self._add_edges_to_collapsed_graph(collapsed_graph, graph)

        return collapsed_graph

    def _calculate_group_sizes(self, graph: Graph) -> None:
        """Create dict with number of nodes in each group."""
        self.group_sizes = {}
        for node in graph.nodes:
            for group in node.groups:
                if not self.group_sizes.get(group):
                    self.group_sizes[group] = 0
                self.group_sizes[group] += 1

    def _add_nodes_to_collapsed_graph(self, collapsed_graph: Graph, graph: Graph) -> None:
        """Add Nodes from source graph to collapsed graph."""
        for node in graph.nodes:
            collapsed_graph.add_node(self._get_node_for_collapsed_graph(node))

    def _add_edges_to_collapsed_graph(self, collapsed_graph: Graph, graph: Graph) -> None:
        """Add Edges from source graph to collapsed graph."""
        collapsed_edges_repository: Dict[str, bool] = {}

        for edge in graph.edges:
            edge_added = collapsed_graph.add_edge(
                source_id=edge.source,
                target_id=edge.target,
            )
            if edge_added:
                continue

            source_node_id = self._get_collapsed_node_id(graph.get_node(edge.source))
            target_node_id = self._get_collapsed_node_id(graph.get_node(edge.target))

            if source_node_id == target_node_id:
                # skip edges inside collapsed node
                continue

            collapsed_edge_designation = (
                f"collapsed edge between {source_node_id} ans {target_node_id}"
            )
            is_collapsed_already_added = collapsed_edges_repository.get(
                collapsed_edge_designation,
                False,
            )
            if not is_collapsed_already_added:
                collapsed_graph.add_edge(
                    source_id=source_node_id,
                    target_id=target_node_id,
                )
                collapsed_edges_repository[collapsed_edge_designation] = True

    def _get_node_for_collapsed_graph(self, node: Node) -> Node:
        """Return a node for collapsed graph."""
        if self._should_collapse_node(node):
            return self._collapse_node(node)
        return node

    def _should_collapse_node(self, node: Node) -> bool:
        """Decide if given node should be collapsed."""
        if not node.groups:
            return False

        for group in node.groups:
            if group not in self.expanded_groups:
                return self.group_sizes.get(group, 0) > 1

        return False

    def _collapse_node(self, node: Node) -> Node:
        """Collapse node into group node."""
        collapsed_node_id = self._get_collapsed_node_id(node)
        group_name = self._unprepare_group_name(collapsed_node_id)
        return GroupNode(id=collapsed_node_id, group_name=group_name)

    def _get_collapsed_node_id(self, node: Node) -> str:
        """Collapse node into group node."""
        if not self._should_collapse_node(node):
            return node.id

        for group in node.groups:
            if group not in self.expanded_groups:
                break

        return self._prepare_group_name(group)

    def _prepare_group_name(self, name: str) -> str:
        """Convert group name to unique one."""
        return f"{self.GROUP_NAME_PREFIX}{name}"

    def _unprepare_group_name(self, name: str) -> str:
        """Convert group name to unique one."""
        if not name.startswith(self.GROUP_NAME_PREFIX):
            return name

        return name[len(self.GROUP_NAME_PREFIX) :]
