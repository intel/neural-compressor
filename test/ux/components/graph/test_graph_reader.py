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
"""Test GraphReader."""

import unittest
from unittest.mock import MagicMock, patch

from lpot.ux.components.graph.graph_reader import GraphReader
from lpot.ux.utils.exceptions import ClientErrorException


class TestGraphReader(unittest.TestCase):
    """Test GraphReader class."""

    @patch("lpot.ux.components.graph.reader.repository.GraphReaderRepository.find")
    def test_ensure_model_readable_fails_for_unknown_framework(
        self,
        mocked_reader_repository_find: MagicMock,
    ) -> None:
        """Test ensure_model_readable fail when not supported framework."""
        mocked_reader_repository_find.side_effect = ClientErrorException

        graph_reader = GraphReader()

        with self.assertRaises(ClientErrorException):
            graph_reader.ensure_model_readable("foo.txt")
        mocked_reader_repository_find.assert_called_once_with("foo.txt")


if __name__ == "__main__":
    unittest.main()
