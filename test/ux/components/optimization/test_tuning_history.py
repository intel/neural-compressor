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
"""Tuning History test."""

import unittest
from unittest.mock import MagicMock, patch

from neural_compressor.ux.components.optimization.tuning_history import tuning_history
from neural_compressor.ux.utils.exceptions import NotFoundException


@patch("sys.argv", ["neural_compressor_bench.py", "-p5000"])
class TestTuningHistory(unittest.TestCase):
    """Test Tuning History."""

    @patch("neural_compressor.ux.utils.templates.workdir.Workdir.get_workload_data")
    def test_get_history_snapshot_when_no_workload_found(
        self,
        mocked_get_workload_data_by_id: MagicMock,
    ) -> None:
        """Test get_history_snapshot."""
        mocked_get_workload_data_by_id.return_value = {}

        with self.assertRaisesRegex(
            NotFoundException,
            "Unable to find workload with id: requested workload id",
        ):
            tuning_history("requested workload id")

        mocked_get_workload_data_by_id.assert_called_with("requested workload id")


if __name__ == "__main__":
    unittest.main()
