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

"""Objects to communicate between domain logic and outside layers."""

from queue import Queue
from typing import Any, Dict, List, Union


class Request:
    """Domain defined parameters."""

    def __init__(self, method: str, operation: str, data: dict) -> None:
        """Initialize properties."""
        self.method: str = method
        self.operation: str = operation
        self.data: dict = data


class Response:
    """Domain defined response."""

    def __init__(self) -> None:
        """Initialize properties."""
        self.data: Union[Dict[str, Any], List[Dict[str, Any]]] = {}
        self.command: Dict[str, Any] = {}


def create_simple_response(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> Response:
    """Create new Response object with only data set."""
    response = Response()
    response.data = data
    return response


class Message:
    """Message used to send data to GUI."""

    def __init__(self, status: str, subject: str, data: Any):
        """Initialize message."""
        self._status: str = status
        self._subject: str = subject
        self._data: Any = data

    @property
    def subject(self) -> str:
        """Get the subject."""
        return self._subject

    @property
    def status(self) -> str:
        """Get the status."""
        return self._status

    @property
    def data(self) -> Any:
        """Get the data."""
        return self._data


class MessageQueue:
    """Queue for passing messages to GUI."""

    _queue: Queue = Queue()

    def __init__(self) -> None:
        """Initialize queue."""
        pass

    def post_failure(self, subject: str, data: Any) -> None:
        """Post failure message."""
        self._queue.put(Message("failure", subject, data))

    def post_success(self, subject: str, data: Any) -> None:
        """Post success message."""
        self._queue.put(Message("success", subject, data))

    def post_error(self, subject: str, data: Any) -> None:
        """Post error message."""
        self._queue.put(Message("error", subject, data))

    def get(self) -> Message:
        """Wait for message and return it."""
        return self._queue.get()
