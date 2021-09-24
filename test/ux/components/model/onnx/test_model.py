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
"""Test Onnx Model."""

import unittest
from unittest.mock import MagicMock, call, patch

from neural_compressor.ux.components.model.onnxrt.model import OnnxrtModel


class TestOnnxrtModel(unittest.TestCase):
    """Test OnnxrtModel class."""

    def test_get_framework_name(self) -> None:
        """Test getting correct framework name."""
        self.assertEqual("onnxrt", OnnxrtModel.get_framework_name())

    def test_supports_correct_path(self) -> None:
        """Test getting correct framework name."""
        self.assertTrue(OnnxrtModel.supports_path("/path/to/model.onnx"))

    def test_supports_incorrect_path(self) -> None:
        """Test getting correct framework name."""
        self.assertFalse(OnnxrtModel.supports_path("/path/to/model.pb"))

    @patch("neural_compressor.ux.components.model.onnxrt.model.check_module")
    def test_guard_requirements_installed(self, mocked_check_module: MagicMock) -> None:
        """Test guard_requirements_installed."""
        model = OnnxrtModel("/path/to/model.onnx")

        model.guard_requirements_installed()

        mocked_check_module.assert_has_calls([call("onnx"), call("onnxruntime")])

    @patch("neural_compressor.ux.components.model.onnxrt.model.check_module")
    def test_get_input_nodes(self, mocked_check_module: MagicMock) -> None:
        """Test getting input nodes."""
        model = OnnxrtModel("/path/to/model.onnx")
        self.assertIsNone(model.get_input_nodes())

    @patch("neural_compressor.ux.components.model.onnxrt.model.check_module")
    def test_get_output_nodes(self, mocked_check_module: MagicMock) -> None:
        """Test getting output nodes."""
        model = OnnxrtModel("/path/to/model.onnx")
        self.assertIsNone(model.get_output_nodes())

    @patch("neural_compressor.ux.components.model.onnxrt.model.check_module")
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
            "neural_compressor.ux.components.model.onnxrt.model.OnnxrtModel class.",
        ):
            OnnxrtModel("/path/to/model.pb")

    def test_get_model_graph(self) -> None:
        """Test getting Graph of a model."""
        with self.assertRaisesRegex(
            NotImplementedError,
            "Reading graph for model /path/to/model.onnx is not supported.",
        ):
            model = OnnxrtModel("/path/to/model.onnx")
            model.get_model_graph()


if __name__ == "__main__":
    unittest.main()
