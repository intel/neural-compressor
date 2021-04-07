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

"""Execution common proc module."""

import datetime
import json
import os
import re
import subprocess
import uuid
from contextlib import ExitStack
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from lpot.ux.utils.logger import log


class Proc(object):
    """Process class that runs commands from shell."""

    def __init__(
        self,
        output_dir: str = ".",
        pid: Optional[str] = None,
        request_id: Optional[str] = None,
        filename: Optional[Union[str, List[str]]] = None,
        additional_log_names: List[str] = [],
    ) -> None:
        """
        Initialize class parameters.

        :param output_dir: path to directory where process output saved
        :param pid unique aibt process identifier
        :return: the constructor returns no value
        """
        pid = pid if pid else uuid.uuid4().hex

        if filename:
            name = filename
        elif request_id:
            name = request_id
        else:
            name = pid
        log_names = [f"{name}.txt"]
        info_file_name = f"{name}.proc"

        log_names.extend(additional_log_names)

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except FileExistsError:
                log.debug("Directory %s already exists.", output_dir)

        # process output file
        self.stage_output_path = os.path.join(output_dir, log_names[0])
        self.log_paths = [os.path.join(output_dir, filename) for filename in log_names]

        # process info file
        self.proc_info_path = os.path.join(output_dir, info_file_name)

        # process information
        self.args: Optional[Iterable[Any]] = None
        self.return_code: Optional[int] = None
        self.time_start: Optional[datetime.datetime] = None
        self.time_stop: Optional[datetime.datetime] = None
        # default ignore exit code is only 0
        self.ignore_exit_codes = [0]

    def run(
        self,
        args: List[Any],
        executable: Optional[Any] = None,
        shell: bool = False,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        universal_newlines: bool = False,
        startupinfo: Optional[Any] = None,
        creationflags: int = 0,
        ignore_exit_codes: Union[list, Any] = None,
    ) -> subprocess.Popen:
        """
        Execute call for process.

        :param args:
        :param executable:
        :param shell:
        :param cwd:
        :param env:
        :param universal_newlines:
        :param startupinfo:
        :param creationflags:
        :param ignore_exit_codes:
        :return: Popen
        """
        self.time_start = datetime.datetime.utcnow()
        self.time_stop = self.time_start
        self.return_code = None
        self.ignore_exit_codes = (
            ignore_exit_codes if ignore_exit_codes and isinstance(ignore_exit_codes, list) else [0]
        )

        try:
            self.args = args
            cmd: Union[str, Any] = " ".join(map(str, args)) if shell else map(str, args)
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=shell,  # nosec
                executable=executable,
                cwd=cwd,
                env=env,
                universal_newlines=universal_newlines,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
            with ExitStack() as stack:
                files = [
                    stack.enter_context(open(fname, "a", encoding="utf-8"))
                    for fname in self.log_paths
                ]
                for line in proc.stdout:  # type: ignore
                    decoded_line = line.decode("utf-8", errors="ignore").strip()
                    log.debug(decoded_line)
                    for log_file in files:
                        log_file.write(decoded_line + "\n")
                        log_file.flush()
            proc.wait()
            self.time_stop = datetime.datetime.utcnow()
            self.return_code = proc.returncode
            msg = "Exit code: {}".format(self.return_code)
            if self.return_code in self.ignore_exit_codes:
                log.debug(msg)
            else:
                log.error(msg)
                log.critical("...")
                for line in self.tail:
                    log.critical(f"\t{line.strip()}")
                relative_output_path = re.sub(
                    r"^.*stages/",
                    "stages/",
                    self.output_path,
                )
                log.critical(f"\tMore in file: {relative_output_path}")
                log.critical("...")

            return proc

        finally:
            self.__save_proc_info()

    def __save_proc_info(self) -> bool:
        """
        Save all proc information in .proc file.

        :raise: bool
        :return: True if save was success and False in all other cases
        """
        if self.time_start and self.time_stop:
            proc_info = {
                "time_start": self.time_start.isoformat(),
                "time_stop": self.time_stop.isoformat(),
                "cmd": self.args,  # list of arguments
                "output_path": self.output_path,
            }

        with open(self.proc_info_path, mode="w") as proc_info_file:
            json.dump(proc_info, proc_info_file, sort_keys=True, indent=4)
            return True
        return False

    @property
    def output(self) -> Iterator[str]:
        """
        Property returns lines form execution.

        :return: Iterator with output lines from process execution
        """
        with open(self.stage_output_path, encoding="utf-8") as log_file:
            for line in log_file:
                yield line

    @property
    def info(self) -> Dict[str, Any]:
        """
        Info contains information about process arguments start and stop time.

        Example:
            "cmd": [
                "docker",
                "pull",
                "hub.docker.intel.com/aibt_tensorflow/centos-7_3_mkldnn_tensorflow:latest"
            ],
            "output_path": "111224/stages/pull_image/0c750a01-90e0-4aa8-99fe-a194d55a2046.txt",
            "time_start": "2018-05-17T14:47:22.352502",
            "time_stop": "2018-05-17T14:47:25.110410"

        :return: Process information
        """
        with open(self.proc_info_path, encoding="utf-8") as proc_info_file:
            return json.load(proc_info_file)

    @property
    def info_path(self) -> str:
        """
        Process .proc file with runtime process information.

        :return: absolute path to process .proc file
        """
        return self.proc_info_path

    @property
    def output_path(self) -> str:
        """
        Process .txt file with process logs.

        :return: absolute path to process .txt file
        """
        if os.path.isfile(self.stage_output_path):
            return self.stage_output_path

        raise Exception("Missing output path in {}".format(self.stage_output_path))

    @property
    def tail(self) -> list:
        """
        Property returns lines form execution along open file.

        :return: Lines from process execution
        """
        lines = 20
        buffer = 4098
        # place holder for the lines found
        lines_found: list = []

        # block counter will be multiplied by buffer
        # to get the block size from the end
        block_counter = -1

        with open(self.output_path, "r") as f:
            # loop until we find X lines
            while len(lines_found) < lines:
                try:
                    f.seek(block_counter * buffer, os.SEEK_END)
                except IOError:  # either file is too small, or too many lines requested
                    f.seek(0)
                    lines_found = f.readlines()
                    break

                lines_found = f.readlines()
                block_counter -= 1

        return lines_found[-lines:]

    @property
    def is_ok(self) -> bool:
        """
        Check if process execute as expected.

        :return: bool if return code from cmd is in ignore_exit_codes return True, False if not
        """
        try:
            return self.return_code in self.ignore_exit_codes
        except Exception:
            return False

    def remove_logs(self) -> None:
        """
        Remove logs generated by process.

        :return: Nothing
        """
        os.remove(self.output_path)
        os.remove(self.info_path)
