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
"""Workdir test."""

import unittest
from copy import deepcopy
from unittest.mock import patch

from lpot.ux.utils.exceptions import InternalException, NotFoundException
from lpot.ux.utils.templates.workdir import Workdir

workloads_dict = {
    "active_workspace_path": "/tmp/lpot",
    "workloads": {
        "workload_1": {
            "id": "workload_1",
            "workload_path": "/tmp/lpot/workload_1",
        },
        "workload_2": {
            "id": "workload_2",
            "workload_path": "/tmp/lpot/workload_2",
        },
    },
}


def _fake_load_mock(self: Workdir) -> dict:
    self.workloads_data = deepcopy(workloads_dict)
    return self.workloads_data


def _fake_raise_exception_mock(*args: tuple, **kwargs: dict) -> None:
    raise Exception("Mocked exception.")


class TestDeleteWorkload(unittest.TestCase):
    """Workdir deletion tests."""

    def test_delete_workload_success(self) -> None:
        """Test workload deletion - success."""
        with patch("os.path.isfile") as fake_isfile, patch(
            "shutil.rmtree",
        ) as fake_rmtree, patch.object(Workdir, "load", autospec=True) as fake_load, patch.object(
            Workdir,
            "dump",
        ) as fake_dump:
            fake_isfile.return_value = True
            fake_rmtree.return_value = None
            fake_load.side_effect = _fake_load_mock
            fake_dump.return_value = None
            workdir = Workdir(workspace_path="/tmp/lpot")
            workdir.delete_workload("workload_1")
            self.assertNotIn("workload_1", workdir.workloads_data.get("workloads", {}))
            self.assertIn("workload_2", workdir.workloads_data.get("workloads", {}))

    def test_delete_workload_missing(self) -> None:
        """Test workload deletion - error due to missing file."""
        with patch("os.path.isfile") as fake_isfile, patch(
            "shutil.rmtree",
        ) as fake_rmtree, patch.object(Workdir, "load", autospec=True) as fake_load, patch.object(
            Workdir,
            "dump",
        ) as fake_dump, self.assertRaises(
            NotFoundException,
        ):
            fake_isfile.return_value = True
            fake_rmtree.return_value = None
            fake_load.side_effect = _fake_load_mock
            fake_dump.return_value = None
            workdir = Workdir(workspace_path="/tmp/lpot")
            workdir.delete_workload("workload_3")

    def test_delete_workload_failure(self) -> None:
        """Test workload deletion - error due to exception."""
        with patch("os.path.isfile") as fake_isfile, patch(
            "shutil.rmtree",
        ) as fake_rmtree, patch.object(Workdir, "load", autospec=True) as fake_load, patch.object(
            Workdir,
            "dump",
        ) as fake_dump, self.assertRaises(
            InternalException,
        ):
            fake_isfile.return_value = True
            fake_rmtree.side_effect = _fake_raise_exception_mock
            fake_load.side_effect = _fake_load_mock
            fake_dump.return_value = None
            workdir = Workdir(workspace_path="/tmp/lpot")
            workdir.delete_workload("workload_1")
