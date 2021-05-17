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
from lpot.ux.utils.workload.workloads_list import WorkloadsListMigrator


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
        self.assertEqual(migrator.require_migration, False)

    def test_migration_v1_to_v2(self) -> None:
        """Test Workloads List Migration from v1 to v2."""
        migrator = WorkloadsListMigrator()
        migrator.workloads_json = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v1.json",
        )
        self.assertEqual(migrator.require_migration, True)
        migrator.migrate()
        self.assertEqual(migrator.require_migration, False)
        expected_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v2.json",
        )
        expected = _load_json_as_dict(expected_json_path)
        self.assertEqual(type(migrator.workloads_data), dict)
        self.assertDictEqual(migrator.workloads_data, expected)  # type: ignore

    def test_not_tuned_migration_v1_to_v2(self) -> None:
        """Test Workloads List Migration from v1 to v2."""
        migrator = WorkloadsListMigrator()
        migrator.workloads_json = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v1_not_tuned.json",
        )
        self.assertEqual(migrator.require_migration, True)
        migrator.migrate()
        self.assertEqual(migrator.require_migration, False)

        expected_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v2_not_tuned.json",
        )
        expected = _load_json_as_dict(expected_json_path)
        self.assertEqual(type(migrator.workloads_data), dict)
        self.assertDictEqual(migrator.workloads_data, expected)  # type: ignore

    def test_empty_migration_v1_to_v2(self) -> None:
        """Test Workloads List Migration from v1 to v2."""
        migrator = WorkloadsListMigrator()
        migrator.workloads_json = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v1_empty.json",
        )
        self.assertEqual(migrator.require_migration, True)
        migrator.migrate()
        self.assertEqual(migrator.require_migration, False)

        expected_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v2_empty.json",
        )
        expected = _load_json_as_dict(expected_json_path)
        self.assertEqual(type(migrator.workloads_data), dict)
        self.assertDictEqual(migrator.workloads_data, expected)  # type: ignore

    def test_migration_v2_to_v2(self) -> None:
        """Test Workloads List Migration from v2 to v2."""
        migrator = WorkloadsListMigrator()
        migrator.workloads_json = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v2.json",
        )
        self.assertEqual(migrator.require_migration, False)
        migrator.migrate()
        self.assertEqual(migrator.require_migration, False)
        expected_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v2.json",
        )
        expected = _load_json_as_dict(expected_json_path)
        self.assertEqual(type(migrator.workloads_data), dict)
        self.assertDictEqual(migrator.workloads_data, expected)  # type: ignore

    def test_not_tuned_migration_v2_to_v2(self) -> None:
        """Test Workloads List Migration from v2 to v2."""
        migrator = WorkloadsListMigrator()
        migrator.workloads_json = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v2_not_tuned.json",
        )
        self.assertEqual(migrator.require_migration, False)
        migrator.migrate()
        self.assertEqual(migrator.require_migration, False)
        expected_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v2_not_tuned.json",
        )
        expected = _load_json_as_dict(expected_json_path)
        self.assertEqual(type(migrator.workloads_data), dict)
        self.assertDictEqual(migrator.workloads_data, expected)  # type: ignore

    def test_empty_migration_v2_to_v2(self) -> None:
        """Test Workloads List Migration from v2 to v2."""
        migrator = WorkloadsListMigrator()
        migrator.workloads_json = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v2_empty.json",
        )
        self.assertEqual(migrator.require_migration, False)
        migrator.migrate()
        self.assertEqual(migrator.require_migration, False)

        expected_json_path = os.path.join(
            os.path.dirname(__file__),
            "files",
            "workloads_list_v2_empty.json",
        )
        expected = _load_json_as_dict(expected_json_path)
        self.assertEqual(type(migrator.workloads_data), dict)
        self.assertDictEqual(migrator.workloads_data, expected)  # type: ignore


if __name__ == "__main__":
    unittest.main()
