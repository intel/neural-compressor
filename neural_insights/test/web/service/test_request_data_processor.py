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
"""RequestDataProcessor test."""

import unittest
from typing import Any, Dict

from neural_insights.utils.exceptions import ClientErrorException
from neural_insights.web.service.request_data_processor import RequestDataProcessor


class TestRequestDataProcessor(unittest.TestCase):
    """Test RequestDataProcessor."""

    def test_get_string_value_with_single_value(self) -> None:
        """Test get_string_value when single value was provided."""
        expected = "expected_value"
        data = {
            "requested_key": [expected],
        }

        self.assertEqual(
            "expected_value",
            RequestDataProcessor.get_string_value(data, "requested_key"),
        )

    def test_get_string_value_with_multiple_values(self) -> None:
        """Test get_string_value when many values were provided."""
        expected = "expected_value"
        data = {
            "requested_key": [
                expected,
                " and ",
                "some",
                "more",
            ],
        }

        self.assertEqual(
            "expected_value",
            RequestDataProcessor.get_string_value(data, "requested_key"),
        )

    def test_get_string_fails_when_data_missing(self) -> None:
        """Test get_string_value fails."""
        data: Dict[str, Any] = {}

        with self.assertRaisesRegex(ClientErrorException, "Missing requested_key parameter"):
            RequestDataProcessor.get_string_value(data, "requested_key")


if __name__ == "__main__":
    unittest.main()
