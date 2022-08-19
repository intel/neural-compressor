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
"""RequestDataProcessor test."""

import unittest
from unittest.mock import MagicMock, patch

from werkzeug.wrappers import Response

from neural_compressor.ux.utils.exceptions import (
    AccessDeniedException,
    ClientErrorException,
    InternalException,
    NotFoundException,
)
from neural_compressor.ux.web.service.response_generator import ResponseGenerator


class TestResponseGenerator(unittest.TestCase):
    """Test ResponseGenerator."""

    @patch("neural_compressor.ux.web.service.response_generator.verify_file_path")
    def test_serve_from_filesystem_passes_verify_file_path_errors(
        self,
        mocked_verify_file_path: MagicMock,
    ) -> None:
        """Test that exceptions from verify_file_path are propagated."""
        mocked_verify_file_path.side_effect = NotFoundException("Expected Error Message")

        with self.assertRaisesRegex(NotFoundException, "Expected Error Message"):
            ResponseGenerator.serve_from_filesystem("foo.txt")

    @patch("neural_compressor.ux.web.service.response_generator.send_file")
    @patch("neural_compressor.ux.web.service.response_generator.verify_file_path")
    def test_serve_from_filesystem(
        self,
        mocked_verify_file_path: MagicMock,
        mocked_send_file: MagicMock,
    ) -> None:
        """Test serve_from_filesystem."""
        expected_value = Response("expected response text")

        mocked_verify_file_path.return_value = None
        mocked_send_file.return_value = expected_value

        actual = ResponseGenerator.serve_from_filesystem("foo.txt", "mimetype")

        self.assertEqual(expected_value, actual)
        mocked_verify_file_path.assert_called_once_with("foo.txt")
        mocked_send_file.assert_called_once_with(
            "foo.txt",
            mimetype="mimetype",
            as_attachment=False,
            max_age=0,
        )

    def test_add_refresh(self) -> None:
        """Test adding refresh header."""
        response = Response()

        self.assertEqual(None, response.headers.get("refresh"))

        ResponseGenerator.add_refresh(response, 15)

        self.assertEqual("15", response.headers.get("refresh"))

    def test_from_exception_for_client_error_exception(self) -> None:
        """Test from_exception for ClientErrorException."""
        message = "Request is invalid!"

        response = ResponseGenerator.from_exception(ClientErrorException(message))

        self.assertEqual(400, response.status_code)
        self.assertEqual(message, response.data.decode("utf-8"))

    def test_from_exception_for_access_denied_exception(self) -> None:
        """Test from_exception for AccessDeniedException."""
        message = "You can't enter here!"

        response = ResponseGenerator.from_exception(AccessDeniedException(message))

        self.assertEqual(403, response.status_code)
        self.assertEqual(message, response.data.decode("utf-8"))

    def test_from_exception_for_not_found_exception(self) -> None:
        """Test from_exception for NotFoundException."""
        message = "There's nothing here!"

        response = ResponseGenerator.from_exception(NotFoundException(message))

        self.assertEqual(404, response.status_code)
        self.assertEqual(message, response.data.decode("utf-8"))

    def test_from_exception_for_internal_exception(self) -> None:
        """Test from_exception for InternalException."""
        message = "Domain code crashed!"

        response = ResponseGenerator.from_exception(InternalException(message))

        self.assertEqual(500, response.status_code)
        self.assertEqual(message, response.data.decode("utf-8"))

    def test_from_exception_for_exception(self) -> None:
        """Test from_exception for Exception."""
        message = "Something crashed!"

        response = ResponseGenerator.from_exception(Exception(message))

        self.assertEqual(500, response.status_code)
        self.assertEqual(message, response.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
