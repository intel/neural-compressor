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
"""Defines _JobsmManager and _Job classes."""
import typing
from collections import OrderedDict
from subprocess import Popen
from threading import Thread
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

from neural_compressor.ux.components.jobs_management.jobs_control_queue import _JobsControlQueue
from neural_compressor.ux.components.jobs_management.request import _Request, _RequestType


class _Job(Thread):
    """Job is a deffered target that may spawn subprocess."""

    _supported_jobs_names = ["benchmark", "optimization", "profiling"]

    def __init__(
        self,
        job_id: str,
        group: None = None,
        target: Optional[Callable[..., object]] = None,
        name: Optional[str] = None,
        args: Optional[Iterable[Any]] = None,
        kwargs: Mapping[str, Any] = {},
        *,
        daemon: Optional[bool] = None,
    ) -> None:
        if args is None:
            args = ()
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self._job_id: str = job_id
        self._subprocess_handle: Optional[Popen] = None
        self.to_be_aborted: bool = False

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def subprocess_handle(self) -> Optional[Popen]:
        return self._subprocess_handle

    @subprocess_handle.setter
    def subprocess_handle(self, subprocess_handle: Popen) -> None:
        if self._subprocess_handle is None:
            self._subprocess_handle = subprocess_handle
        else:
            raise Exception(
                f"Job subprocess handle can be assign only onced.\
                Current value {self._subprocess_handle}.",
            )

    @staticmethod
    def parse_job_id(job_name: str, name_id: Union[int, str]) -> str:
        name_id = int(name_id)
        if job_name not in _Job._supported_jobs_names:
            raise Exception(
                f"Job name shuld be one of {_Job._supported_jobs_names}. \
            Got {job_name}.",
            )
        return f"{job_name}_{name_id}"


class _JobsManager:
    """Oparates on request_queue, recives tasks and processes them."""

    def __init__(self, request_queue: _JobsControlQueue, daemon: bool = True):
        self._request_queue: _JobsControlQueue = request_queue
        self._jobs: typing.OrderedDict[str, _Job] = OrderedDict()
        self._main_thread: Thread = Thread(target=self.main_loop, daemon=daemon)
        self._request_methods: Dict[_RequestType, Callable] = {
            _RequestType.SCHEDULE: self._schedule_job,
            _RequestType.ABORT: self._abort_job,
            _RequestType.ADD_PROCESS_HANDLE: self._add_process_handle,
            _RequestType.DELETE_JOB: self._delete_job,
        }

    def start(self) -> None:
        if not self._main_thread.is_alive():
            self._main_thread.start()

    def main_loop(self) -> None:
        while True:
            request = self._request_queue._get()
            self.process_request(request)

    def process_request(self, request: _Request) -> None:
        method = self._request_methods.get(request.type, None)
        if method is None:
            raise Exception(f"Not recognized task type: {request.type}.")
        method(request)

    def _schedule_job(self, request: _Request) -> None:
        # there already is job with such job_id
        if self._jobs.get(request.job_id, None):
            raise Exception(f"Logic error. Job with this job_id exists {request.job_id}.")
        else:
            job = _Job(request.job_id, target=request.target, args=request.args)
            self._jobs[job.job_id] = job
            if len(self._jobs) == 1:
                self._start_first()

    def _abort_job(self, request: _Request) -> None:
        job = self._jobs.get(request.job_id, None)
        # job has already been terminated or never created
        if job is None:
            # do nothing
            pass
        # job already spawned subprocess
        elif job.is_alive() and job.subprocess_handle is not None:
            # if there is subprocess terminate it, it it has exited already,
            # it has no impact on handle
            job.subprocess_handle.terminate()

        # job is alive and has not spawned subprocess
        elif job.is_alive() and job.subprocess_handle is None:
            # if job spawns subprocess it will be terminated
            job.to_be_aborted = True

        # job is in jobs queue and it is not alive and does not have subprocess
        # so it can be deleted from queue
        elif not job.is_alive() and job.subprocess_handle is None:
            del self._jobs[job.job_id]

        # if request has event it means that it is synchronous request waiting for job to end
        if request.event is not None:
            if job is not None and job.is_alive():
                job.join()
            request.event.set()

        # in other case job is not alive but still exists on jobs queue,
        # so wait for proper job manager cleanup in _end_job,
        # it happens when end notifiacation of this job has not been processed yet,
        # but abort was recived;
        # one error indicating case is not checked when there is subprocess but job has finished
        return

    def _add_process_handle(self, request: _Request) -> None:
        """Add process handle to job object on jobs queue."""
        job = self._jobs.get(request.job_id, None)
        # check for current assumption that only one job runs at a time,
        # remove this check if running multiple jobs simultaneously is allowed
        if job is not self._get_first():
            raise Exception(
                "Logic error.\
                 Try to add subprocess to job different than first in queue.",
            )
        if job is None:
            # because of delay in handling jobs requests,
            # job may finish before its' add_subprocess is handled
            return
        if job.subprocess_handle is not None:
            raise Exception("Logic error. Try to add another subprocess to a job.")

        if job.is_alive():
            if request.process_handle is None:
                raise TypeError("Process handle missing.")
            job.subprocess_handle = request.process_handle
            if job.to_be_aborted:
                job.subprocess_handle.terminate()

    def _delete_job(self, request: _Request) -> None:
        """Delete job from the jobs queue."""
        job = self._jobs.get(request.job_id, None)
        if job is None:
            raise Exception("Processing job end notification but job doest not exist.")
        # check if selected job is first in queue,
        # remove this check if jobs' parallelism is allowed
        if job != self._get_first():
            raise Exception("Tries to delete job that is not first in queue.")
        del self._jobs[request.job_id]
        self._start_first()

    def _start_first(self) -> None:
        """Start first job in queue, in case of empty queue do nothing."""
        first_job = self._get_first()
        if first_job is None:
            # there is no more jobs in queue
            return
        if not first_job.is_alive():
            first_job.start()
        else:
            raise Exception("Logic error, first job is already running.")

    def _get_first(self) -> Optional[_Job]:
        """Return reference to first job in queue, none in case of empty queue."""
        try:
            first = next(iter(self._jobs.values()))
        except StopIteration:
            first = None
        return first
