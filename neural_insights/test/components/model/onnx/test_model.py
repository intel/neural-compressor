# -*- coding: utf-8 -*-
# Copyright (c) 2023 Intel Corporation
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
"""Test Onnx Model."""

import unittest
from unittest.mock import MagicMock, call, patch

from neural_insights.components.graph.graph import Graph
from neural_insights.components.model.onnxrt.model import (
    OnnxrtModel,
    remove_number_of_samples_from_shape,
)
from neural_insights.utils.consts import Frameworks


class TestOnnxrtModel(unittest.TestCase):
    """Test OnnxrtModel class."""

    def test_get_framework_name(self) -> None:
        """Test getting correct framework name."""
        self.assertEqual(Frameworks.ONNX.value, OnnxrtModel.get_framework_name())

    def test_supports_correct_path(self) -> None:
        """Test getting correct framework name."""
        self.assertTrue(OnnxrtModel.supports_path("/path/to/model.onnx"))

    def test_supports_incorrect_path(self) -> None:
        """Test getting correct framework name."""
        self.assertFalse(OnnxrtModel.supports_path("/path/to/model.pb"))

    @patch("neural_insights.components.model.onnxrt.model.check_module")
    def test_guard_requirements_installed(self, mocked_check_module: MagicMock) -> None:
        """Test guard_requirements_installed."""
        model = OnnxrtModel("/path/to/model.onnx")

        model.guard_requirements_installed()

        mocked_check_module.assert_has_calls([call("onnx"), call("onnxruntime")])

    @patch("neural_insights.components.model.onnxrt.model.check_module")
    def test_get_input_nodes(self, mocked_check_module: MagicMock) -> None:
        """Test getting input nodes."""
        model = OnnxrtModel("/path/to/model.onnx")
        self.assertIsNone(model.get_input_nodes())

    @patch("neural_insights.components.model.onnxrt.model.check_module")
    def test_get_output_nodes(self, mocked_check_module: MagicMock) -> None:
        """Test getting output nodes."""
        model = OnnxrtModel("/path/to/model.onnx")
        self.assertIsNone(model.get_output_nodes())

    @patch("neural_insights.components.model.onnxrt.model.check_module")
    def test_get_input_and_output_nodes(self, mocked_check_module: MagicMock) -> None:
        """Test getting input nodes."""
        model = OnnxrtModel("/path/to/model.onnx")
        self.assertIsNone(model.get_input_nodes())
        self.assertIsNone(model.get_output_nodes())

    def test_ensure_supported_path(self) -> None:
        """Test ensure_supported_path."""
        with self.assertRaisesRegex(
            AttributeError,
            "Model path: /path/to/model.pb is not supported by "
            "neural_insights.components.model.onnxrt.model.OnnxrtModel class.",
        ):
            OnnxrtModel("/path/to/model.pb")

    @patch(
        "neural_insights.components.model.onnxrt.model.OnnxrtReader",
        autospec=True,
    )
    def test_get_model_graph(self, mocked_onnxrt_graph_reader: MagicMock) -> None:
        """Test getting Graph of a model."""
        expected = Graph()

        mocked_onnxrt_graph_reader.return_value.read.return_value = expected

        model = OnnxrtModel("/path/to/model.onnx")

        self.assertEqual(expected, model.get_model_graph())

        mocked_onnxrt_graph_reader.assert_called_once_with(model)

    def test_shape_elements_order(self) -> None:
        """Test getting shape elements order."""
        model = OnnxrtModel("/path/to/model.onnx")
        self.assertListEqual(model.shape_elements_order, ["channels", "height", "width"])

    def test_remove_number_of_samples_from_shape(self) -> None:
        """Test removing number of samples from shape."""
        shape = "1,3,224,224"
        actual = remove_number_of_samples_from_shape(shape)

        expected = "3,224,224"
        self.assertEqual(expected, actual)

    def test_remove_number_of_samples_from_standard_shape(self) -> None:
        """Test removing number of samples from shape."""
        shape = "1,2,3"
        actual = remove_number_of_samples_from_shape(shape)

        expected = "1,2,3"
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
