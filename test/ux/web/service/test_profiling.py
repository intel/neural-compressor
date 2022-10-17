# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
"""ProfilingService test."""

import unittest
from unittest.mock import MagicMock, patch

from werkzeug.wrappers import Response

from neural_compressor.ux.utils.exceptions import ClientErrorException, NotFoundException
from neural_compressor.ux.web.service.profiling import ProfilingService


class TestProfilingService(unittest.TestCase):
    """Test ProfilingService."""

    def test_get_config_fails_when_no_workload_id_requested(self) -> None:
        """Test get_config."""
        with self.assertRaisesRegex(ClientErrorException, "Missing id parameter"):
            ProfilingService.get_config({})

    @patch("neural_compressor.ux.web.service.profiling.ProfilingService._get_workload_data")
    def test_get_config_fails_when_no_workload_found(
        self,
        mocked_get_workload_data: MagicMock,
    ) -> None:
        """Test get_config."""
        mocked_get_workload_data.return_value = {}

        with self.assertRaisesRegex(NotFoundException, "Unable to find config file"):
            ProfilingService.get_config(
                {
                    "id": [1],
                },
            )

        mocked_get_workload_data.assert_called_with({"id": [1]})

    @patch("neural_compressor.ux.web.service.profiling.ProfilingService._get_workload_data")
    def test_get_config_fails_when_config_path_missing(
        self,
        mocked_get_workload_data: MagicMock,
    ) -> None:
        """Test get_config."""
        mocked_get_workload_data.return_value = {
            "config_path": None,
        }

        with self.assertRaisesRegex(
            NotFoundException,
            "Unable to find config file",
        ):
            ProfilingService.get_config(
                {
                    "id": [1],
                },
            )

        mocked_get_workload_data.assert_called_with({"id": [1]})

    @patch("neural_compressor.ux.web.service.workload.ResponseGenerator.serve_from_filesystem")
    @patch("neural_compressor.ux.web.service.profiling.ProfilingService._get_workload_data")
    def test_get_config(
        self,
        mocked_get_workload_data: MagicMock,
        mocked_serve_from_filesystem: MagicMock,
    ) -> None:
        """Test get_config."""
        mocked_get_workload_data.return_value = {
            "config_path": "/some/fake/config/path.yaml",
        }
        expected = Response("fake config content")

        mocked_serve_from_filesystem.return_value = expected

        actual = ProfilingService.get_config(
            {
                "id": [1],
            },
        )

        self.assertEqual(expected, actual)
        mocked_get_workload_data.assert_called_with({"id": [1]})
        mocked_serve_from_filesystem.assert_called_once_with(
            path="/some/fake/config/path.yaml",
            mimetype="text/vnd.yaml",
        )

    def test_get_code_template_fails_when_no_workload_id_requested(self) -> None:
        """Test get_code_template."""
        with self.assertRaisesRegex(ClientErrorException, "Missing id parameter"):
            ProfilingService.get_code_template({})

    @patch("neural_compressor.ux.web.service.profiling.ProfilingService._get_workload_data")
    def test_get_code_template_fails_when_no_workload_found(
        self,
        mocked_get_workload_data: MagicMock,
    ) -> None:
        """Test get_code_template."""
        mocked_get_workload_data.return_value = {}

        with self.assertRaisesRegex(
            NotFoundException,
            "Unable to find code template file",
        ):
            ProfilingService.get_code_template(
                {
                    "id": [1],
                },
            )

        mocked_get_workload_data.assert_called_with({"id": [1]})

    @patch("neural_compressor.ux.web.service.profiling.ProfilingService._get_workload_data")
    def test_get_code_template_fails_when_code_template_path_missing(
        self,
        mocked_get_workload_data: MagicMock,
    ) -> None:
        """Test get_code_template."""
        mocked_get_workload_data.return_value = {
            "code_template_path": None,
        }

        with self.assertRaisesRegex(
            NotFoundException,
            "Unable to find code template file",
        ):
            ProfilingService.get_code_template(
                {
                    "id": [1],
                },
            )

        mocked_get_workload_data.assert_called_with({"id": [1]})

    @patch("neural_compressor.ux.web.service.workload.ResponseGenerator.serve_from_filesystem")
    @patch("neural_compressor.ux.web.service.profiling.ProfilingService._get_workload_data")
    def test_get_code_template(
        self,
        mocked_get_workload_data: MagicMock,
        mocked_serve_from_filesystem: MagicMock,
    ) -> None:
        """Test get_code_template."""
        mocked_get_workload_data.return_value = {
            "code_template_path": "/some/fake/code/template/path.py",
        }
        expected = Response("fake code template content")

        mocked_serve_from_filesystem.return_value = expected

        actual = ProfilingService.get_code_template(
            {
                "id": [1],
            },
        )

        self.assertEqual(expected, actual)
        mocked_get_workload_data.assert_called_with({"id": [1]})
        mocked_serve_from_filesystem.assert_called_once_with(
            path="/some/fake/code/template/path.py",
            mimetype="text/x-python",
        )

    def test_get_output_fails_when_no_workload_id_requested(self) -> None:
        """Test get_output."""
        with self.assertRaisesRegex(ClientErrorException, "Missing id parameter"):
            ProfilingService.get_output({})

    @patch("neural_compressor.ux.web.service.profiling.ProfilingService._get_workload_data")
    def test_get_output_fails_when_no_workload_found(
        self,
        mocked_get_workload_data: MagicMock,
    ) -> None:
        """Test get_output."""
        mocked_get_workload_data.return_value = {}

        with self.assertRaisesRegex(
            NotFoundException,
            "Unable to find output log",
        ):
            ProfilingService.get_output(
                {
                    "id": [1],
                },
            )

        mocked_get_workload_data.assert_called_with({"id": [1]})

    @patch("neural_compressor.ux.web.service.profiling.ProfilingService._get_workload_data")
    def test_get_output_fails_when_log_path_missing(
        self,
        mocked_get_workload_data: MagicMock,
    ) -> None:
        """Test get_output."""
        mocked_get_workload_data.return_value = {
            "log_path": None,
        }

        with self.assertRaisesRegex(
            NotFoundException,
            "Unable to find output log",
        ):
            ProfilingService.get_output(
                {
                    "id": [1],
                },
            )

        mocked_get_workload_data.assert_called_with({"id": [1]})

    @patch("neural_compressor.ux.web.service.workload.ResponseGenerator.serve_from_filesystem")
    @patch("neural_compressor.ux.web.service.profiling.ProfilingService._get_workload_data")
    def test_get_output(
        self,
        mocked_get_workload_data: MagicMock,
        mocked_serve_from_filesystem: MagicMock,
    ) -> None:
        """Test get_output."""
        data = {
            "id": [1],
            "autorefresh": [15],
        }
        filesystem_response = Response("fake output content")

        mocked_get_workload_data.return_value = {
            "log_path": "/some/fake/output/path.log",
        }
        mocked_serve_from_filesystem.return_value = filesystem_response

        actual = ProfilingService.get_output(data)

        self.assertEqual(filesystem_response.data, actual.data)
        self.assertEqual("15", actual.headers.get("refresh"))
        mocked_get_workload_data.assert_called_with(data)
        mocked_serve_from_filesystem.assert_called_once_with(
            path="/some/fake/output/path.log",
            mimetype="text/plain",
        )

    @patch("neural_compressor.ux.web.service.workload.ResponseGenerator.serve_from_filesystem")
    @patch("neural_compressor.ux.web.service.profiling.ProfilingService._get_workload_data")
    def test_get_output_with_failure(
        self,
        mocked_get_workload_data: MagicMock,
        mocked_serve_from_filesystem: MagicMock,
    ) -> None:
        """Test get_output."""
        data = {
            "id": [1],
        }
        mocked_get_workload_data.return_value = {
            "log_path": "/some/fake/output/path.log",
        }
        mocked_serve_from_filesystem.side_effect = NotFoundException("Unable to find file.")

        actual = ProfilingService.get_output(data)

        self.assertEqual("Unable to find file.", actual.data.decode("utf-8"))
        self.assertEqual(404, actual.status_code)
        self.assertIsNone(actual.headers.get("refresh", None))
        mocked_get_workload_data.assert_called_with(data)
        mocked_serve_from_filesystem.assert_called_once_with(
            path="/some/fake/output/path.log",
            mimetype="text/plain",
        )


if __name__ == "__main__":
    unittest.main()
