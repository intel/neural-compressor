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
"""Workloads list test."""

import os
import unittest

from lpot.ux.utils.utils import _load_json_as_dict
from lpot.ux.utils.workload.workload import WorkloadMigrator


class TestWorkloadsListMigrator(unittest.TestCase):
    """Workloads list migrator tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Workloads list migrator test constructor."""
        super().__init__(*args, **kwargs)

    def test_workload_migration_from_v1(self) -> None:
        """Test Workload v1 config migrator."""
        workload_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workload_v1_tuned.json",
        )
        workload_migrator = WorkloadMigrator(
            workload_json_path=workload_json_path,
        )
        workload_migrator.migrate()

        print(workload_migrator.workload_data)

        expected_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workload_v2_tuned.json",
        )
        expected = _load_json_as_dict(expected_json_path)
        self.assertDictEqual(workload_migrator.workload_data, expected)  # type: ignore

    def test_workload_migration_from_v2(self) -> None:
        """Test Workload v2 config migrator."""
        workload_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workload_v2_tuned.json",
        )
        workload_migrator = WorkloadMigrator(
            workload_json_path=workload_json_path,
        )
        workload_migrator.migrate()

        expected_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workload_v2_tuned.json",
        )
        expected = _load_json_as_dict(expected_json_path)
        self.assertDictEqual(workload_migrator.workload_data, expected)  # type: ignore


if __name__ == "__main__":
    unittest.main()
