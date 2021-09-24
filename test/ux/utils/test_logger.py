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
"""Logger test."""

import logging
import unittest

from neural_compressor.ux.utils.logger import change_log_level, log


class TestLogger(unittest.TestCase):
    """Logger tests."""

    def test_changing_log_level(self) -> None:
        """Test default values."""
        change_log_level(logging.INFO)
        self.assertEqual(logging.INFO, log.level)


if __name__ == "__main__":
    unittest.main()
