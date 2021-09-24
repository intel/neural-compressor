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
"""Tensorflow Graph reader."""

from typing import Dict, List, Optional

from tensorflow.core.framework.attr_value_pb2 import AttrValue
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.python.framework.dtypes import _TYPE_TO_STRING

from neural_compressor.ux.components.graph.attribute import Attribute
from neural_compressor.ux.components.graph.graph import Graph
from neural_compressor.ux.components.graph.node import Node
from neural_compressor.ux.components.model.model import Model
from neural_compressor.ux.utils.exceptions import NotFoundException


class TensorflowReader:
    """Graph Reader for Tensorflow."""

    def __init__(self, model: Model) -> None:
        """Initialize object."""
        self._hidden_node_ids: Dict[str, bool] = {}
        self.model: Model = model

    def read(self) -> Graph:
        """Read a graph."""
        self._hidden_node_ids = {}

        from neural_compressor.ux.components.model.tensorflow.model import TensorflowModel

        if not isinstance(self.model, TensorflowModel):
            raise NotFoundException(f"{self.model.path} is not Tensorflow model.")

        graph_def = self.model.nc_model_instance.graph_def

        graph = Graph()

        for node_def in graph_def.node:
            if self._should_hide_node(node_def):
                self._hide_node(node_def)
                continue

            current_node_id = node_def.name

            graph.add_node(
                Node(
                    id=current_node_id,
                    label=node_def.op,
                    properties={
                        "name": node_def.name,
                        "type": node_def.op,
                    },
                    attributes=self._convert_attributes(node_def),
                    groups=self._get_group_names(current_node_id),
                ),
            )

            for input_node_name in node_def.input:
                if self._is_node_id_hidden(input_node_name) or self._is_node_id_hidden(
                    current_node_id,
                ):
                    continue
                graph.add_edge(
                    source_id=input_node_name,
                    target_id=current_node_id,
                )

        return graph

    def _get_group_names(self, node_id: str) -> List[str]:
        """Get a group names from Node starting from the top."""
        elements = node_id.split("/")
        names = []
        for idx in range(1, len(elements)):
            names.append("/".join(elements[0:idx]))
        return names

    def _should_hide_node(self, node_def: NodeDef) -> bool:
        """Check if given node should be hidden."""
        if node_def.op not in ["Const", "Identity"]:
            return False

        if self._node_def_has_not_hidden_inputs(node_def):
            return False

        return True

    def _node_def_has_not_hidden_inputs(self, node_def: NodeDef) -> bool:
        """Check if node has any visible input."""
        not_hidden_input_ids = list(
            filter(
                lambda input_node: input_node not in self._hidden_node_ids,
                node_def.input,
            ),
        )
        return len(not_hidden_input_ids) > 0

    def _hide_node(self, node: NodeDef) -> None:
        """Mark node as hidden."""
        self._hidden_node_ids[node.name] = True

    def _is_node_id_hidden(self, node_id: str) -> bool:
        """Check if provided node_id is hidden."""
        return node_id in self._hidden_node_ids

    def _convert_attributes(self, node_def: NodeDef) -> List[Attribute]:
        """Convert NodeDef attributes to our format."""
        attributes = []

        for attribute_name in node_def.attr:
            converted_attribute = self._convert_attribute(
                attribute_name,
                node_def.attr[attribute_name],
            )
            if converted_attribute is not None:
                attributes.append(converted_attribute)

        return attributes

    def _convert_attribute(self, name: str, value: AttrValue) -> Optional[Attribute]:
        """Convert NodeDef attribute to our format."""
        if 0 != len(value.s):
            return Attribute(name, "string", str(value.s.decode("utf-8")))
        if 0 != value.type:
            return Attribute(
                name,
                "type",
                _TYPE_TO_STRING.get(value.type, "UNKNOWN"),
            )
        return None
