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

"""Definition of executor."""

import datetime
from threading import Lock, Thread
from typing import Any, List, Optional, Union

from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.proc import Proc
from neural_compressor.ux.utils.processes import NCProcesses
from neural_compressor.ux.web.communication import MessageQueue

LOCK = Lock()


class Executor:
    """Executor class provide execute shell command."""

    def __init__(
        self,
        workspace_path: str,
        subject: str,
        data: Optional[dict] = None,
        send_response: bool = True,
        log_name: Optional[str] = None,
        additional_log_names: List[str] = [],
    ) -> None:
        """
        Bash Executor constructor.

        :param args: arguments for parser execution
        """
        if data:
            self._request_id = str(data.get("id", ""))
        self._log_name = log_name
        self._additional_log_names = additional_log_names
        self._workdir = workspace_path
        self._mq = MessageQueue()
        self._subject = subject
        self._send_response = send_response
        self.data = data
        self.time_start: Optional[datetime.datetime] = None
        self.time_stop: Optional[datetime.datetime] = None

    def call_one(
        self,
        args: List[Any],
        logger: Optional[Any] = None,
        executable: Optional[Any] = None,
        shell: bool = False,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        universal_newlines: bool = False,
        startupinfo: Optional[Any] = None,
        creationflags: int = 0,
        processes: Optional[Any] = None,
        ignore_exit_codes: Union[list, Any] = None,
        pid: Optional[str] = None,
    ) -> None:
        """
        Execute single call for process.

        :param args:
        :param logger:
        :param executable:
        :param shell:
        :param cwd:
        :param env:
        :param universal_newlines:
        :param startupinfo:
        :param creationflags:
        :param processes:
        :param ignore_exit_codes:
        :param pid:
        """
        proc = None

        try:
            log.debug("Exec %s ", " ".join(map(str, args)))
            proc = Proc(
                self.workdir,
                pid=pid,
                request_id=self.request_id,
                filename=self.log_name,
                additional_log_names=self.additional_log_names,
            )
            if self._send_response:
                self._mq.post_success(
                    "_".join([self._subject, "start"]),
                    self.data,
                )
            proc.run(
                args,
                executable,
                shell,
                cwd,
                env,
                universal_newlines,
                startupinfo,
                creationflags,
                ignore_exit_codes,
            )
            self.time_start = proc.time_start
            self.time_stop = proc.time_stop

        except Exception:
            if self._send_response:
                self._mq.post_error(
                    self._subject,
                    {"message": Exception, "id": self.request_id},
                )
            log.exception("Unexpected error for command line %s", args)
        try:
            LOCK.acquire()
            if processes is None:
                processes = []
            processes.append(proc)
        finally:
            LOCK.release()
            log.debug("DONE")

    def call(
        self,
        args: List[Any],  # type: ignore
        logger: Optional[Any] = None,
        executable: Optional[Any] = None,
        shell: bool = False,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        universal_newlines: bool = False,
        startupinfo: Optional[Any] = None,
        creationflags: int = 0,
        env_args: Optional[list] = None,
        ignore_exit_codes: Union[list, Any] = None,
        pid: Optional[str] = None,
    ) -> NCProcesses:
        """
        Execute multiple calls for process.

        :param args:
        :param logger:
        :param executable:
        :param shell:
        :param cwd:
        :param env:
        :param universal_newlines:
        :param startupinfo:
        :param creationflags:
        :param ignore_exit_codes:
        :param pid:
        """
        # update output directory to current stage
        self.refresh_workdir()

        threads = []
        processes = NCProcesses()

        if not self.is_multi_commands(args):
            args = [args]

        for arg in args:
            threads.append(
                Thread(
                    target=self.call_one,
                    args=(
                        arg,
                        logger,
                        executable,
                        shell,
                        cwd,
                        env_args,
                        universal_newlines,
                        startupinfo,
                        creationflags,
                        processes,
                        ignore_exit_codes,
                        pid,
                    ),
                    daemon=True,
                ),
            )

        # Start all threads
        for command_thread in threads:
            command_thread.start()

        # Wait for all of them to finish
        for command_thread in threads:
            command_thread.join()

        return processes

    @property
    def process_duration(self) -> Optional[float]:
        """Return duration of the process in [s]."""
        if self.time_start and self.time_stop:
            duration = self.time_stop - self.time_start
            return duration.total_seconds()
        return None

    @property
    def workdir(self) -> str:
        """
        Property concat workdir path.

        :return: workdir path
        """
        return self._workdir

    @property
    def request_id(self) -> Optional[str]:
        """
        Property contain info about request.

        :return: request id
        """
        return self._request_id

    @property
    def log_name(self) -> Optional[str]:
        """
        Property contain info about output file name.

        :return: requested log filename
        """
        return self._log_name

    @property
    def additional_log_names(self) -> List[str]:
        """
        Property contain info about additional output file names.

        :return: list of additional log filenames
        """
        return self._additional_log_names

    def refresh_workdir(self) -> str:
        """
        Property returns workdir method.

        :return: workdir path
        """
        return self.workdir

    @staticmethod
    def is_multi_commands(args: list) -> bool:
        """
        Check type of execution.

        :param args: arguments for parser execution
        :return: bool value True if args are in list else False
        """
        for arg in args:
            if not isinstance(arg, list):
                return False
        # all elements must be lists
        return True
