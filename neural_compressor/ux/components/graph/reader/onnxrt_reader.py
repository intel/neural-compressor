# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
"""Onnxrt Graph reader."""

from typing import Dict, List, Optional

from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.onnx_ml_pb2 import AttributeProto, NodeProto

from neural_compressor.ux.components.graph.attribute import Attribute
from neural_compressor.ux.components.graph.graph import Graph
from neural_compressor.ux.components.graph.node import Node
from neural_compressor.ux.components.model.model import Model
from neural_compressor.ux.utils.exceptions import NotFoundException


class OnnxrtReader:
    """Graph Reader for Onnxrt."""

    def __init__(self, model: Model) -> None:
        """Initialize object."""
        self._hidden_node_ids: Dict[str, bool] = {}
        self.model: Model = model

    def read(self) -> Graph:
        """Read a graph."""
        self._hidden_node_ids = {}

        from neural_compressor.ux.components.model.onnxrt.model import OnnxrtModel

        if not isinstance(self.model, OnnxrtModel):
            raise NotFoundException(f"{self.model.path} is not Onnxrt model.")

        onnx_nodes = self.model.nc_model_instance.nodes()
        graph = Graph()

        graph = self._add_boundary_nodes(graph)

        edges: dict = {}

        for node_def in onnx_nodes:
            if self._should_hide_node(node_def):
                self._hide_node(node_def)
                continue

            edges = self._add_edges_from_node(edges, node_def)

            current_node_id = node_def.name

            graph.add_node(
                Node(
                    id=current_node_id,
                    label=node_def.op_type,
                    highlight=False,
                    properties={
                        "name": node_def.name,
                        "type": node_def.op_type,
                    },
                    attributes=self._convert_attributes(node_def),
                    groups=self._get_group_names(current_node_id),
                ),
            )

        self._process_edges(graph, edges)

        return graph

    def _get_group_names(self, node_id: str) -> List[str]:
        """Get a group names from Node starting from the top."""
        elements = node_id.split("/")
        names = []
        for idx in range(1, len(elements)):
            names.append("/".join(elements[0:idx]))
        return names

    def _should_hide_node(self, node_def: NodeProto) -> bool:
        """Check if given node should be hidden."""
        if node_def.op_type not in ["Const", "Identity"]:
            return False

        if self._node_def_has_not_hidden_inputs(node_def):
            return False

        return True

    def _node_def_has_not_hidden_inputs(self, node_def: NodeProto) -> bool:
        """Check if node has any visible input."""
        not_hidden_input_ids = list(
            filter(
                lambda input_node: input_node not in self._hidden_node_ids,
                node_def.input,
            ),
        )
        return len(not_hidden_input_ids) > 0

    def _hide_node(self, node: NodeProto) -> None:
        """Mark node as hidden."""
        self._hidden_node_ids[node.name] = True

    def _is_node_id_hidden(self, node_id: str) -> bool:
        """Check if provided node_id is hidden."""
        return node_id in self._hidden_node_ids

    def _convert_attributes(self, node_def: NodeProto) -> List[Attribute]:
        """Convert NodeDef attributes to our format."""
        attributes = []

        for attribute in node_def.attribute:
            converted_attribute = self._convert_attribute(attribute)
            if converted_attribute is not None:
                attributes.append(converted_attribute)

        return attributes

    def _convert_attribute(self, attribute: AttributeProto) -> Optional[Attribute]:
        """Convert NodeDef attribute to our format."""
        if 0 != len(attribute.s):
            return Attribute(attribute.name, "string", str(attribute.s.decode("utf-8")))
        if 0 != attribute.type:
            return Attribute(
                attribute.name,
                "type",
                str(TENSOR_TYPE_TO_NP_TYPE.get(attribute.type, "UNKNOWN")),
            )
        return None

    def _add_boundary_nodes(self, graph: Graph) -> Graph:
        """Add boundary nodes to graph."""
        from neural_compressor.ux.components.model.onnxrt.model import OnnxrtModel

        if not isinstance(self.model, OnnxrtModel):
            raise NotFoundException(f"{self.model.path} is not Onnxrt model.")

        onnx_graph = self.model.nc_model_instance.graph()
        boundary_nodes = {"Input node": onnx_graph.input, "Output node": onnx_graph.output}
        for node_type, nodes in boundary_nodes.items():
            for node in nodes:
                graph.add_node(
                    Node(
                        id=node.name,
                        label=node_type,
                        highlight=False,
                        properties={
                            "name": node.name,
                            "type": node_type,
                        },
                        attributes=[],
                        groups=self._get_group_names(node.name),
                    ),
                )
        return graph

    def _add_edges_from_node(self, edges: dict, node: NodeProto) -> dict:
        """Add update edges from node data."""
        for edge in node.input:
            edge_data = edges.get(edge, {})
            input_nodes = edge_data.get("inputs", [])
            input_nodes.append(node.name)
            edge_data.update({"inputs": input_nodes})
            edges.update({edge: edge_data})

        for edge in node.output:
            edge_data = edges.get(edge, {})
            output_nodes = edge_data.get("outputs", [])
            output_nodes.append(node.name)
            edge_data.update({"outputs": output_nodes})
            edges.update({edge: edge_data})

        return edges

    def _process_edges(self, graph: Graph, edges: dict) -> None:
        """Process edges."""
        from neural_compressor.ux.components.model.onnxrt.model import OnnxrtModel

        if not isinstance(self.model, OnnxrtModel):
            raise NotFoundException(f"{self.model.path} is not Onnxrt model.")

        for edge, edge_data in edges.items():
            # Check if edge connects to input node
            input_nodes = [node.name for node in self.model.nc_model_instance.graph().input]
            if edge in input_nodes and "inputs" in edge_data:
                source_node = edge
                for target_node in edge_data["inputs"]:
                    graph.add_edge(
                        source_id=source_node,
                        target_id=target_node,
                    )

            # Check if edge connects to output node
            output_nodes = [node.name for node in self.model.nc_model_instance.graph().output]
            if edge in output_nodes and "outputs" in edge_data:
                target_node = edge
                for source_node in edge_data["outputs"]:
                    graph.add_edge(
                        source_id=source_node,
                        target_id=target_node,
                    )

            self._process_standard_edge(graph, edge_data)

    def _process_standard_edge(self, graph: Graph, edge_data: dict) -> None:
        """Process standard edges that contain both inputs and outputs."""
        if "inputs" not in edge_data or "outputs" not in edge_data:
            return
        for source_node in edge_data["outputs"]:
            for target_node in edge_data["inputs"]:
                if self._is_node_id_hidden(source_node) or self._is_node_id_hidden(
                    target_node,
                ):
                    continue
                graph.add_edge(
                    source_id=source_node,
                    target_id=target_node,
                )
