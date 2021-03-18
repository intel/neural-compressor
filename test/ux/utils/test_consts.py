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
"""Consts test."""

import unittest

from lpot.ux.utils.consts import github_info


class TestConsts(unittest.TestCase):
    """Consts tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Consts tests constructor."""
        super().__init__(*args, **kwargs)

    def test_github_info(self) -> None:
        """Test if path is correctly recognized as hidden."""
        self.assertIs(type(github_info), dict)
        self.assertEqual(github_info.get("user"), "intel")
        self.assertEqual(github_info.get("repository"), "lpot")
        self.assertEqual(github_info.get("tag"), "v1.2")


if __name__ == "__main__":
    unittest.main()
