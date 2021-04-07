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


if __name__ == "__main__":
    unittest.main()
