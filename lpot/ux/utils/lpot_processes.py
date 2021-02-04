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

"""Execution common process module."""


class LPOTProcesses(list):
    """
    Processes class aggregates Process list.

    Provide helper methods to retrieve information about all executed processes.
    """

    def return_code_all(self) -> None:
        """
        Provide list of return codes of all Process.

        :rtype : list
        :return: List of int with process return codes.
        """
        return_codes = []
        for lpot_process in self:
            return_codes.append(lpot_process.return_code)

    @property
    def is_ok(self) -> bool:
        """
        Property provide information if all executed process during one call executed properly.

        :rtype : bool
        :return: return False if at least one of processes failed, in other case return True
        """
        for lpot_process in self:
            if not lpot_process.is_ok:
                return False
        return True

    def __str__(self) -> str:
        """
        Provide list of information about all executed process.

        :rtype : list
        :return: list of processes args and int with process return codes
        """
        Lpot_processes = []
        for lpot_process in self:
            Lpot_processes.append(
                "LpotProc(\n\tcmd={}\n\treturn_code={}".format(
                    " ".join(map(str, lpot_process.args)),
                    lpot_process.return_code,
                ),
            )
        return "\n".join(Lpot_processes)

    def remove_successful_logs(self) -> None:
        """Remove call logs if all statuses are successful."""
        if self.is_ok:
            for process in self:
                process.remove_logs()
