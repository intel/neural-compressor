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
"""Test Node."""

import unittest
import uuid

from neural_compressor.ux.components.graph.node import Node


def _get_random_string() -> str:
    """Create random string to be used."""
    return uuid.uuid4().hex


class TestNode(unittest.TestCase):
    """Test Node class."""

    def test_setting_parameters(self) -> None:
        """Test if parameters are correctly set."""
        id = _get_random_string()
        label = _get_random_string()

        node = Node(id=id, label=label)

        self.assertEqual(id, node.id)
        self.assertEqual(label, node.label)


if __name__ == "__main__":
    unittest.main()
