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
"""Neural Insights Communication test."""

import unittest

from neural_insights.web.communication import (
    Message,
    MessageQueue,
    Request,
    Response,
    create_simple_response,
)


class TestCommunication(unittest.TestCase):
    """Neural Insights Communication tests."""

    def setUp(self) -> None:
        """Create test environment."""
        self.queue = MessageQueue()

    def test_request(self) -> None:
        """Test that Request is working."""
        method = "GET"
        operation = "/api/a/b/x"
        data = self._get_random_dict()
        request = Request(method, operation, data)

        self.assertEqual(method, request.method)
        self.assertEqual(operation, request.operation)
        self.assertEqual(data, request.data)

    def test_response(self) -> None:
        """Test that Response is working."""
        response = Response()

        self.assertEqual({}, response.data)
        self.assertEqual({}, response.command)

    def test_create_simple_response(self) -> None:
        """Test that create_simple_response is working."""
        data = self._get_random_dict()
        response = create_simple_response(data)

        self.assertEqual(data, response.data)
        self.assertEqual({}, response.command)

    def test_message(self) -> None:
        """Test that Message is working."""
        status = "Test status"
        subject = "Test subject"
        data = self._get_random_dict()
        message = Message(status, subject, data)

        self.assertEqual(status, message.status)
        self.assertEqual(subject, message.subject)
        self.assertEqual(data, message.data)

    def test_message_queue_post_failure(self) -> None:
        """Test posting failure messages to message queue."""
        data = self._get_random_dict()
        self.queue.post_failure("subject", data)
        self._assert_message("failure", "subject", data)

    def test_message_queue_post_success(self) -> None:
        """Test posting success messages to message queue."""
        data = self._get_random_dict()
        self.queue.post_success("subject", data)
        self._assert_message("success", "subject", data)

    def test_message_queue_post_error(self) -> None:
        """Test posting error messages to message queue."""
        data = self._get_random_dict()
        self.queue.post_error("subject", data)
        self._assert_message("error", "subject", data)

    def _get_random_dict(self, size: int = 5) -> dict:
        """Build random dict."""
        from numpy.random import randint

        return {"key " + str(i): randint(65536) for i in range(size)}

    def _assert_message(
        self,
        expected_status: str,
        expected_subject: str,
        expected_data: dict,
    ) -> None:
        """Assert message in queue matches expectations."""
        message = self.queue.get()

        self.assertEqual(expected_status, message.status)
        self.assertEqual(expected_subject, message.subject)
        self.assertEqual(expected_data, message.data)


if __name__ == "__main__":
    unittest.main()
