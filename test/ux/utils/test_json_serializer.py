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
"""Json Serializer test."""

import unittest
from typing import Any, List

from lpot.ux.utils.json_serializer import JsonSerializer


class SubClassToTest(JsonSerializer):
    """Test sub class for json serializer tests."""

    def __init__(self) -> None:
        """Initialize test class."""
        super().__init__()
        self.some_variable: str = "value"


class ClassToTest(JsonSerializer):
    """Test class for json serializer tests."""

    def __init__(self) -> None:
        """Initialize test class."""
        super().__init__()
        self._skip.append("skip_test")
        self.skip_test: str = "this should be skipped"
        self.none_value: Any = None
        self.string: str = "some string"
        self.empty_list: List = []
        self.some_list: List[Any] = ["a", 1]
        self._private_var: int = 1
        self.sub = SubClassToTest()
        subclass1 = SubClassToTest()
        subclass1.some_variable = "this is subclass1 variable"
        subclass2 = SubClassToTest()
        subclass2.some_variable = "this is subclass2 variable"
        self.sub_list: List[SubClassToTest] = [subclass1, subclass2]
        self.empty_dict: dict = {}
        self.dict_with_empty_values = {
            "foo": None,
            "bar": None,
        }
        self.dict_with_some_empty_values = {
            "foo": None,
            "bar": 12,
            "baz": None,
            "donk": 42,
        }


class TestJsonSerializer(unittest.TestCase):
    """Json serializer tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Parser tests constructor."""
        super().__init__(*args, **kwargs)

    def test_serialization(self) -> None:
        """Test if path is correctly recognized as hidden."""
        test_object = ClassToTest()
        result = test_object.serialize()
        expected = {
            "string": "some string",
            "some_list": ["a", 1],
            "private_var": 1,
            "sub": {
                "some_variable": "value",
            },
            "sub_list": [
                {
                    "some_variable": "this is subclass1 variable",
                },
                {
                    "some_variable": "this is subclass2 variable",
                },
            ],
            "dict_with_some_empty_values": {
                "bar": 12,
                "donk": 42,
            },
        }
        self.assertEqual(type(result), dict)
        self.assertDictEqual(result, expected)  # type: ignore


if __name__ == "__main__":
    unittest.main()
