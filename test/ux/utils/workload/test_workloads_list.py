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

from neural_compressor.ux.utils.utils import _load_json_as_dict
from neural_compressor.ux.utils.workload.workloads_list import WorkloadsListMigrator


class TestWorkloadsListMigrator(unittest.TestCase):
    """Workloads list migrator tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Workloads list migrator test constructor."""
        super().__init__(*args, **kwargs)

    def test_clean_environment_migration(self) -> None:
        """Test migration of clean environment."""
        migrator = WorkloadsListMigrator()
        migrator.workloads_json = os.path.join(
            os.path.dirname(__file__),
            "files",
            "non_existing_workloads_list.json",
        )
        self.assertFalse(migrator.require_migration)

    def test_migration_from_v1(self) -> None:
        """Test Workloads List Migration from v1."""
        self._assert_migrates(
            "workloads_list_v1.json",
            "workloads_list_v3.json",
        )

    def test_not_tuned_migration_from_v1(self) -> None:
        """Test Workloads List Migration from v1."""
        self._assert_migrates(
            "workloads_list_v1_not_tuned.json",
            "workloads_list_v3_not_tuned.json",
        )

    def test_empty_migration_from_v1(self) -> None:
        """Test Workloads List Migration from v1."""
        self._assert_migrates(
            "workloads_list_v1_empty.json",
            "workloads_list_v3_empty.json",
        )

    def test_migration_from_v2(self) -> None:
        """Test Workloads List Migration from v2."""
        self._assert_migrates(
            "workloads_list_v2.json",
            "workloads_list_v3.json",
        )

    def test_not_tuned_migration_from_v2(self) -> None:
        """Test Workloads List Migration from v2."""
        self._assert_migrates(
            "workloads_list_v2_not_tuned.json",
            "workloads_list_v3_not_tuned.json",
        )

    def test_empty_migration_from_v2(self) -> None:
        """Test Workloads List Migration from v2."""
        self._assert_migrates(
            "workloads_list_v2_empty.json",
            "workloads_list_v3_empty.json",
        )

    def test_migration_from_v3(self) -> None:
        """Test Workloads List Migration from v3."""
        self._assert_doesnt_migrate("workloads_list_v3.json")

    def test_not_tuned_migration_from_v3(self) -> None:
        """Test Workloads List Migration from v3."""
        self._assert_doesnt_migrate("workloads_list_v3_not_tuned.json")

    def test_empty_migration_from_v3(self) -> None:
        """Test Workloads List Migration from v3."""
        self._assert_doesnt_migrate("workloads_list_v3_empty.json")

    def _assert_migrates(self, initial_file: str, final_file: str) -> None:
        """Test Workloads List Migration from initial to final version."""
        self._assert_migration_works(initial_file, final_file, True)

    def _assert_doesnt_migrate(self, initial_file: str) -> None:
        """Test Workloads List Migration doesn't migrate final version."""
        self._assert_migration_works(initial_file, initial_file, False)

    def _assert_migration_works(
        self,
        initial_file: str,
        final_file: str,
        migration_needed: bool,
    ) -> None:
        """Test Workloads List Migration from initial to final version."""
        migrator = WorkloadsListMigrator()
        migrator.workloads_json = os.path.join(
            os.path.dirname(__file__),
            "files",
            initial_file,
        )

        self.assertEqual(migrator.require_migration, migration_needed)

        migrator.migrate()

        self.assertFalse(migrator.require_migration)

        expected_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            final_file,
        )
        expected = _load_json_as_dict(expected_json_path)
        self.assertEqual(type(migrator.workloads_data), dict)
        self.assertDictEqual(migrator.workloads_data, expected)  # type: ignore


if __name__ == "__main__":
    unittest.main()
