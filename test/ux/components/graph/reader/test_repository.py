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
"""Test GraphReaderRepository class."""
import unittest
from unittest.mock import MagicMock, patch

from lpot.ux.components.graph.reader.repository import GraphReaderRepository
from lpot.ux.utils.exceptions import ClientErrorException


class GraphReaderRepositoryTestCase(unittest.TestCase):
    """Test GraphReaderRepository class."""

    def setUp(self) -> None:
        """Create test env."""
        self.repository = GraphReaderRepository()

    def test_getting_reader_of_not_model_file_fails(self) -> None:
        """Test if reading of not model file fails."""
        with self.assertRaises(ClientErrorException):
            self.repository.find("not_a_model.pdf")

    def test_getting_reader_of_not_tensorflow_model_file_fails(self) -> None:
        """Test if reading of not model file fails."""
        with self.assertRaises(ClientErrorException):
            self.repository.find("not_a_tensorflow_model.onnx")

    def test_getting_reader_of_pb_model_file_when_reader_not_registered_fails(
        self,
    ) -> None:
        """Test if reading of not model file fails."""
        self.repository._framework_readers = {}
        with self.assertRaises(ClientErrorException):
            self.repository.find("model.pb")

    @patch("lpot.ux.components.graph.reader.repository.get_framework_from_path")
    def test_getting_reader_for_frozen_pb(self, mocked_get_framework_from_path: MagicMock) -> None:
        """Test getting reader for frozen_pb file."""
        try:
            from lpot.ux.components.graph.reader.tensorflow import TensorflowReader

            mocked_get_framework_from_path.return_value = "tensorflow"

            reader = self.repository.find("/path/to/frozen_pb.pb")

            self.assertIsInstance(reader, TensorflowReader)
        except ModuleNotFoundError:
            self.skipTest("Skipping reading graph of frozen_pb as Tensorflow is not installed.")


if __name__ == "__main__":
    unittest.main()
