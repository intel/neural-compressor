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
"""Neural Insights Router test."""

import os
import shutil
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from neural_insights.web.communication import Request
from neural_insights.web.exceptions import ServiceNotFoundException
from neural_insights.web.router import Router

MOCKED_HOME = os.path.abspath(os.path.dirname(__file__))


@patch.dict(os.environ, {"HOME": MOCKED_HOME})
class TestRouter(unittest.TestCase):
    """Neural Insights Router tests."""

    def setUp(self) -> None:
        """Prepare environment."""
        assert not os.path.isdir(self._get_workdir_path())
        self._paths_to_clean = [self._get_workdir_path()]

    def tearDown(self) -> None:
        """Make environment clean again."""
        for path in self._paths_to_clean:
            shutil.rmtree(path, ignore_errors=True)

    def test_handle_fails_on_unknown_path(self) -> None:
        """Test that passing unknown path fails."""
        request = Request("GET", "/foo/bar", {})
        router = Router()

        with self.assertRaises(ServiceNotFoundException):
            router.handle(request)

    @patch("neural_insights.web.router.get_workloads_list")
    def test_executes_expected_realtime_function(
        self,
        mocked_get_workloads: MagicMock,
    ) -> None:
        """Test that passing known realtime endpoint succeeds."""
        expected_return = {
            "workloads": [
                {
                    "accuracy_data": {
                        "baseline_accuracy": 0.7664,
                        "optimized_accuracy": 0.7634,
                        "ratio": -0.003914405010438417,
                    },
                    "creation_time": 1685605003,
                    "framework": "ONNXRT",
                    "mode": "quantization",
                    "model_path": "/some/path/to/model.onnx",
                    "status": "success",
                    "uuid": "d34ec577-0b84-44d3-9e31-66ad884c5759",
                    "workload_location": "/path/to/nc_workspace/2023-06-01_09-36-39",
                },
                {
                    "creation_time": 1685620648,
                    "framework": "ONNXRT",
                    "mode": "benchmark",
                    "model_path": "/some/path/to/model.onnx",
                    "status": "success",
                    "uuid": "d51b6028-c6c8-4227-b8e9-6f9b46845ab5",
                    "workload_location": "/path/to/nc_workspace/2023-06-01_09-36-39",
                },
            ],
        }
        mocked_get_workloads.return_value = expected_return
        request = Request("GET", "workloads", {})

        router = Router()
        response = router.handle(request)

        self.assertEqual({}, response.command)
        self.assertEqual(expected_return, response.data)

    def _get_workdir_path(self) -> str:
        """Build local path to workdir."""
        return os.path.join(MOCKED_HOME, ".neural_compressor")

    def _mocked_workdir_init(self, *args: Any, **kwargs: Any) -> None:
        self.assertEqual(MOCKED_HOME, kwargs.get("workspace_path"))


if __name__ == "__main__":
    unittest.main()
