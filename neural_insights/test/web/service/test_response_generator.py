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
"""RequestDataProcessor test."""

import unittest

from werkzeug.wrappers import Response

from neural_insights.utils.exceptions import (
    AccessDeniedException,
    ClientErrorException,
    InternalException,
    NotFoundException,
)
from neural_insights.web.service.response_generator import ResponseGenerator


class TestResponseGenerator(unittest.TestCase):
    """Test ResponseGenerator."""

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
