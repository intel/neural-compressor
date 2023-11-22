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
"""Utils test."""

import unittest
from unittest.mock import MagicMock, patch

from neural_insights.utils.consts import Frameworks
from neural_insights.utils.exceptions import (
    ClientErrorException,
)
from neural_insights.utils.utils import (
    check_module,
    get_file_extension,
    get_framework_from_path,
)

fake_metrics: dict = {
    "topk": {},
    "COCOmAP": {},
    "MSE": {},
    "RMSE": {},
    "MAE": {},
    "metric1": {},
}


class TestUtils(unittest.TestCase):
    """Value parser tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Parser tests constructor."""
        super().__init__(*args, **kwargs)

    @patch("neural_insights.components.model.tensorflow.frozen_pb.get_model_type")
    def test_get_tensorflow_framework_from_path(self, mocked_get_model_type: MagicMock) -> None:
        """Test getting framework name from path."""
        mocked_get_model_type.return_value = "frozen_pb"
        path = "/home/user/model.pb"
        result = get_framework_from_path(path)
        self.assertEqual(result, Frameworks.TF.value)
        mocked_get_model_type.assert_called_with(path)

    def test_get_onnx_framework_from_path(self) -> None:
        """Test getting framework name from path."""
        path = "/home/user/model.onnx"
        result = get_framework_from_path(path)
        self.assertEqual(result, Frameworks.ONNX.value)

    def test_get_unknown_framework_from_path(self) -> None:
        """Test getting framework name from path."""
        path = "/home/user/model.some_extension"
        result = get_framework_from_path(path)
        self.assertIsNone(result)

    def test_get_file_extension(self) -> None:
        """Test getting file extension from path."""
        path = "/home/user/file.ext"
        result = get_file_extension(path)
        self.assertEqual(result, "ext")

    def test_get_file_with_dots_extension(self) -> None:
        """Test getting file extension from path."""
        path = "/home/user/file.name.ext2"
        result = get_file_extension(path)
        self.assertEqual(result, "ext2")

    def test_get_file_without_extension(self) -> None:
        """Test getting file extension from path."""
        path = "/home/user/file"
        result = get_file_extension(path)
        self.assertEqual(result, "")

    def test_check_module(self) -> None:
        """Test checking existing module."""
        check_module("os")

    def test_check_non_existing_module(self) -> None:
        """Test checking non existing module."""
        with self.assertRaises(ClientErrorException):
            check_module("non_existing_module")


if __name__ == "__main__":
    unittest.main()
