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
"""UX Router test."""

import os
import shutil
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from lpot.ux.utils.templates.workdir import Workdir
from lpot.ux.web.communication import Request
from lpot.ux.web.exceptions import ServiceNotFoundException
from lpot.ux.web.router import Router

MOCKED_HOME = os.path.abspath(os.path.dirname(__file__))


@patch.dict(os.environ, {"HOME": MOCKED_HOME})
class TestRouter(unittest.TestCase):
    """UX Router tests."""

    def setUp(self) -> None:
        """Prepare environment."""
        assert not os.path.isdir(self._get_workdir_path())
        self._paths_to_clean = [self._get_workdir_path()]
        mocked_workdir_init = patch.object(Workdir, "__init__", self._mocked_workdir_init)
        mocked_workdir_init.__enter__()
        self._workdir_clean_status_patcher = patch.object(Workdir, "clean_status")
        self._mocked_workdir_clean_status = self._workdir_clean_status_patcher.start()

    def tearDown(self) -> None:
        """Make environment clean again."""
        self._workdir_clean_status_patcher.stop()
        for path in self._paths_to_clean:
            shutil.rmtree(path, ignore_errors=True)

    def test_it_cleans_wip_status(self) -> None:
        """Test that WIP status of existing workloads is cleaned."""
        Router()
        self._mocked_workdir_clean_status.assert_called_once_with(status_to_clean="wip")

    def test_handle_fails_on_unknown_path(self) -> None:
        """Test that passing unknown path fails."""
        request = Request("GET", "/foo/bar", {})
        router = Router()

        with self.assertRaises(ServiceNotFoundException):
            router.handle(request)

    @patch("lpot.ux.web.router.get_directory_entries")
    def test_executes_expected_realtime_function(
        self,
        mocked_get_directory_entries: MagicMock,
    ) -> None:
        """Test that passing known realtime endpoint succeeds."""
        expected_return = {
            "foo": "bar",
        }
        mocked_get_directory_entries.return_value = expected_return
        request = Request("GET", "filesystem", {"path": ["."]})

        router = Router()
        response = router.handle(request)

        self.assertEqual({}, response.command)
        self.assertEqual(expected_return, response.data)

    @patch("lpot.ux.web.router.get_boundary_nodes")
    @patch("lpot.ux.web.router.Thread")
    def test_executes_expected_deferred_function(
        self,
        mocked_thread: MagicMock,
        mocked_function: MagicMock,
    ) -> None:
        """Test that passing known deferred endpoint succeeds."""
        params = {
            "foo": "bar",
        }
        request = Request("GET", "get_boundary_nodes", params)

        router = Router()
        response = router.handle(request)

        mocked_thread.assert_called_once_with(target=mocked_function, args=(params,))

        self.assertEqual({}, response.command)
        self.assertEqual({"exit_code": 102, "message": "processing"}, response.data)

    def test_handle_fails_on_unsupported_route_definition_type(self) -> None:
        """Test that not supported route definition type fails."""
        from lpot.ux.components.file_browser.file_browser import get_directory_entries
        from lpot.ux.web.router import RoutingDefinition

        router = Router()
        router.routes = {
            "foo": RoutingDefinition(get_directory_entries),
        }
        request = Request("GET", "foo", {})

        with self.assertRaises(ValueError):
            router.handle(request)

    def _get_workdir_path(self) -> str:
        """Build local path to workdir."""
        return os.path.join(MOCKED_HOME, ".lpot")

    def _mocked_workdir_init(self, *args: Any, **kwargs: Any) -> None:
        self.assertEqual(MOCKED_HOME, kwargs.get("workspace_path"))


if __name__ == "__main__":
    unittest.main()
