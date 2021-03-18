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
"""Model config test."""

import unittest
from typing import Any, List

from lpot.ux.utils.workload.model import Model, parse_nodes


class TestModelConfig(unittest.TestCase):
    """Model config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Model config constructor."""
        super().__init__(*args, **kwargs)

    def test_parse_none_nodes(self) -> None:
        """Test parsing None to empty list."""
        nodes = None
        result = parse_nodes(nodes)
        self.assertEqual(result, [])

    def test_parse_string_nodes(self) -> None:
        """Test parsing string to empty list."""
        nodes = "input"
        result = parse_nodes(nodes)
        self.assertEqual(result, ["input"])

    def test_parse_nodes_empty_list(self) -> None:
        """Test parsing list to list."""
        nodes: List[Any] = []
        result = parse_nodes(nodes)
        self.assertEqual(result, [])

    def test_parse_nodes_list(self) -> None:
        """Test parsing nodes list to list."""
        nodes = ["input_1", "input_2"]
        result = parse_nodes(nodes)
        self.assertEqual(result, ["input_1", "input_2"])

    def test_string_single_nodes_constructor(self) -> None:
        """Test if path is correctly recognized as hidden."""
        model_config = Model(
            {
                "name": "model_name",
                "framework": "framework_name",
                "inputs": "input",
                "outputs": "predict",
            },
        )
        self.assertEqual(model_config.inputs, "input")
        self.assertEqual(model_config.outputs, "predict")

    def test_string_multiple_nodes_constructor(self) -> None:
        """Test if path is correctly recognized as hidden."""
        model_config = Model(
            {
                "name": "model_name",
                "framework": "framework_name",
                "inputs": "input, another_input",
                "outputs": "predict, softmax",
            },
        )
        self.assertEqual(model_config.inputs, "input, another_input")
        self.assertEqual(model_config.outputs, "predict, softmax")

    def test_string_missing_nodes_constructor(self) -> None:
        """Test if path is correctly recognized as hidden."""
        model_config = Model(
            {
                "name": "model_name",
                "framework": "framework_name",
            },
        )
        self.assertEqual(model_config.inputs, [])
        self.assertEqual(model_config.outputs, [])

    def test_inputs_setter(self) -> None:
        """Test model config inputs setter."""
        model_config = Model(
            {
                "name": "model_name",
                "framework": "framework_name",
            },
        )
        self.assertEqual(model_config.inputs, [])
        model_config.inputs = "input"
        self.assertEqual(model_config.inputs, ["input"])

    def test_outputs_setter(self) -> None:
        """Test model config outputs setter."""
        model_config = Model(
            {
                "name": "model_name",
                "framework": "framework_name",
            },
        )
        self.assertEqual(model_config.outputs, [])
        model_config.outputs = ["softmax", "predict"]
        self.assertEqual(model_config.outputs, ["softmax", "predict"])

    def test_model_without_nodes_serialization(self) -> None:
        """Test model config serialization."""
        model_config = Model(
            {
                "name": "model_name",
                "framework": "framework_name",
            },
        )
        result = model_config.serialize()
        expected = {
            "name": "model_name",
            "framework": "framework_name",
        }
        self.assertDictEqual(result, expected)

    def test_model_single_node_serialization(self) -> None:
        """Test model config serialization."""
        model_config = Model(
            {
                "name": "model_name",
                "framework": "framework_name",
                "inputs": "input",
                "outputs": "predict",
            },
        )
        result = model_config.serialize()
        expected = {
            "name": "model_name",
            "framework": "framework_name",
            "inputs": "input",
            "outputs": "predict",
        }
        self.assertDictEqual(result, expected)

    def test_model_multiple_nodes_serialization(self) -> None:
        """Test model config serialization."""
        model_config = Model(
            {
                "name": "model_name",
                "framework": "framework_name",
                "inputs": ["input"],
                "outputs": ["predict", "softmax"],
            },
        )
        result = model_config.serialize()
        expected = {
            "name": "model_name",
            "framework": "framework_name",
            "inputs": "input",
            "outputs": ["predict", "softmax"],
        }
        self.assertDictEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
