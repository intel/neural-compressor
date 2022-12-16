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
"""Defines _JobsControlQueue class for queuing requests to JobsManager."""
from queue import Queue
from subprocess import Popen
from threading import Event
from typing import Callable, Optional, Tuple

from neural_compressor.ux.components.jobs_management.request import _Request, _RequestType
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.logger import log


class _JobsControlQueue:
    """Class for queuing requests controlling deffered jobs."""

    _requests: Queue = Queue()

    def schedule_job(self, target: Callable, args: Tuple, job_id: str, request_id: str) -> None:
        """Schedule new job to be run."""
        wrapped_target = self._wrap_target(target, job_id)
        task = _Request(
            _RequestType.SCHEDULE,
            target=wrapped_target,
            args=args,
            job_id=job_id,
            request_id=request_id,
        )
        self._requests.put(task)

    def abort_job(
        self,
        job_id: str,
        request_id: Optional[str] = None,
        blocking: bool = False,
    ) -> None:
        """Put request to kill running job."""
        event = None
        if blocking:
            event = Event()

        task = _Request(
            _RequestType.ABORT,
            request_id=request_id,
            job_id=job_id,
            event=event,
        )
        self._requests.put(task)

        if event:
            event.wait()

    def add_subprocess(self, job_id: str, process_handle: Popen) -> None:
        """Add subprocess to currently executed job."""
        task = _Request(
            _RequestType.ADD_PROCESS_HANDLE,
            process_handle=process_handle,
            job_id=job_id,
        )
        self._requests.put(task)

    def _get(self) -> _Request:
        """Wait for request and return it."""
        return self._requests.get()

    def _wrap_target(self, target: Callable, job_id: str) -> Callable:
        """Add putting information that thread finished executing target to the task queue."""

        def wraped_target(args: Tuple) -> None:
            try:
                target(args)
            except ClientErrorException as e:
                log.error(e)
            finally:
                request = _Request(_RequestType.DELETE_JOB, job_id=job_id)
                self._requests.put(request)

        return wraped_target
