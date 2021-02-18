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
"""Configuration model module."""

from typing import Any, Dict, List, Union

from lpot.ux.utils.json_serializer import JsonSerializer


class Model(JsonSerializer):
    """Configuration Model class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize configuration Model class."""
        super().__init__()
        self.name = data.get("name", None)
        self.framework = data.get("framework", {})
        self._inputs: List[str] = data.get("inputs", [])
        self._outputs: List[str] = data.get("outputs", [])

    @property
    def inputs(self) -> List[str]:
        """Get inputs."""
        return self._inputs

    @inputs.setter
    def inputs(self, value: Union[None, str, List[str]]) -> None:
        """Set inputs value."""
        self._inputs = parse_nodes(value)

    @property
    def outputs(self) -> List[str]:
        """Get outputs."""
        return self._outputs

    @outputs.setter
    def outputs(self, value: Union[None, str, List[str]]) -> None:
        """Set outputs value."""
        self._outputs = parse_nodes(value)

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize configuration Model class."""
        item = {
            "name": self.name,
            "framework": self.framework,
        }
        if len(self.inputs) > 0:
            if len(self.inputs) == 1:
                item.update({"inputs": self.inputs[0]})
            else:
                item.update({"inputs": self.inputs})
        if len(self.outputs) > 0:
            if len(self.outputs) == 1:
                item.update({"outputs": self.outputs[0]})
            else:
                item.update({"outputs": self.outputs})
        return item


def parse_nodes(nodes: Union[None, str, List[str]]) -> List[str]:
    """Parse boundary nodes to required format."""
    if isinstance(nodes, str):
        return [nodes]
    if isinstance(nodes, list):
        return nodes
    return []
