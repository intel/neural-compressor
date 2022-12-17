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
"""Test jobs control queue."""

import unittest
import unittest.mock
import uuid
from copy import deepcopy
from subprocess import Popen
from threading import Thread
from typing import Tuple

import numpy as np

from neural_compressor.ux.components.jobs_management.jobs_control_queue import _JobsControlQueue
from neural_compressor.ux.components.jobs_management.request import _Request, _RequestType

TIMEOUT_PERIOD = 2
SCALAR = 5


def _get_random_string() -> str:
    """Create random string to be used."""
    return uuid.uuid4().hex


class TestJobsControlQueue(unittest.TestCase):
    """Test _JobsControlQueue class."""

    def setUp(self) -> None:
        """Create test environment."""
        self.queue = _JobsControlQueue()
        self.job_id = _get_random_string()
        self.request_id = _get_random_string()

    def test_wrap_target(self) -> None:
        """Test wrapping target to add delete request to queue."""

        def target(_tuple: Tuple):
            _tuple[0].append(SCALAR)

        wrapped_target = self.queue._wrap_target(target, self.job_id)
        starting_list = list(np.random.randint(1000, size=10))
        target_list = deepcopy(starting_list)
        wrapped_target((target_list,))
        target((starting_list,))
        self.assertEqual(
            target_list,
            starting_list,
            "Target was not executed.",
        )
        request = self.queue._get()
        self.assertRequestEqual(
            request,
            type=_RequestType.DELETE_JOB,
            job_id=self.job_id,
        )

    def test_schedule_job(self):
        """Test adding schedule request to queue."""

        def target():
            return -1

        starting_list = list(np.random.randint(1000, size=10))

        self.queue.schedule_job(
            target,
            (starting_list,),
            self.job_id,
            self.request_id,
        )
        request = self.queue._get()
        self.assertRequestEqual(
            request,
            _RequestType.SCHEDULE,
            job_id=self.job_id,
            request_id=self.request_id,
            args=(starting_list,),
        )

    def test_abort_job(self):
        """Test adding abort job request to queue."""
        self.queue.abort_job(
            self.job_id,
            self.request_id,
        )
        request = self.queue._get()

        self.assertRequestEqual(
            request,
            _RequestType.ABORT,
            job_id=self.job_id,
            request_id=self.request_id,
        )

    def test_abort_job_blocking(self):
        """Test blocking flag in abort_job method."""
        adding_abort = Thread(
            target=self.queue.abort_job,
            args=(
                self.job_id,
                self.request_id,
                True,
            ),
            daemon=True,
        )
        adding_abort.start()
        abort_request = self.queue._get()
        self.assertTrue(adding_abort.is_alive())
        abort_request.event.set()
        adding_abort.join(timeout=TIMEOUT_PERIOD)
        self.assertFalse(
            adding_abort.is_alive(),
            "Abort job thread did not finish.",
        )

    def test_abort_job_non_blocking(self):
        """Test non_blocking adding abort job request to queue."""
        adding_abort = Thread(
            target=self.queue.abort_job,
            args=(
                self.job_id,
                self.request_id,
            ),
            daemon=True,
        )
        adding_abort.start()
        self.queue._get()  # remove element from queue for next tests
        adding_abort.join(timeout=TIMEOUT_PERIOD)
        self.assertFalse(
            adding_abort.is_alive(),
            "Abort job thread did not finish.",
        )

    def test_add_subprocess(self):
        """Test adding add_subprocess request to queue."""
        process_handle = unittest.mock.Mock(Popen)
        self.queue.add_subprocess(
            self.job_id,
            process_handle,
        )
        request = self.queue._get()
        self.assertRequestEqual(
            request,
            _RequestType.ADD_PROCESS_HANDLE,
            job_id=self.job_id,
            process_handle=process_handle,
        )

    def assertRequestEqual(
        self,
        request: _Request,
        type: _RequestType,
        job_id: str = None,
        request_id: str = None,
        process_handle: Popen = None,
        args: Tuple = None,
        is_event: bool = False,
    ):
        """Unified assert for request fields testing."""
        self.assertEqual(
            request.type,
            type,
            "Request type is incorrect.",
        )
        self.assertEqual(
            request.job_id,
            job_id,
            "Job id is incorrect.",
        )
        if request_id:
            self.assertEqual(
                request.request_id,
                request_id,
                "Request id is incorrect.",
            )
        if process_handle:
            self.assertIs(
                request.process_handle,
                process_handle,
                "Process handle is incorrect.",
            )
        if args:
            self.assertEqual(
                request.args,
                args,
                "Arguments are incorrect.",
            )
        if is_event:
            self.assertIsNotNone(
                request.event,
            )
        else:
            self.assertIsNone(
                request.event,
            )


if __name__ == "__main__":
    unittest.main()
