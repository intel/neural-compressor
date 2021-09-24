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
"""Utils test."""

import unittest
from unittest.mock import MagicMock

import yaml

from neural_compressor.ux.utils.yaml_utils import float_representer


class TestYamlUtils(unittest.TestCase):
    """Yaml Utils tests."""

    def test_represent_int(self) -> None:
        """Test representing int."""
        expected = 123.0

        dumper_mock = MagicMock(yaml.Dumper)
        dumper_mock.represent_scalar.return_value = expected

        actual = float_representer(dumper_mock, float(123))

        self.assertEqual(expected, actual)
        dumper_mock.represent_scalar.assert_called_once_with("tag:yaml.org,2002:float", "123.0")

    def test_represent_float(self) -> None:
        """Test representing int."""
        expected = 123.56

        dumper_mock = MagicMock(yaml.Dumper)
        dumper_mock.represent_scalar.return_value = expected

        actual = float_representer(dumper_mock, 123.56)

        self.assertEqual(expected, actual)
        dumper_mock.represent_scalar.assert_called_once_with("tag:yaml.org,2002:float", "123.56")


if __name__ == "__main__":
    unittest.main()
