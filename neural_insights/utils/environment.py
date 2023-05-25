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
"""Environment manager class."""
import os
import sys


class Environment:
    """Environment manager class."""

    @staticmethod
    def ensure_workdir_exists_and_writeable() -> None:
        """Ensure that configured directory exists and can be used."""
        from neural_insights.utils.logger import log
        from neural_insights.web.configuration import Configuration

        configuration = Configuration()
        workdir = configuration.workdir
        error_message_tail = "Please ensure it is a directory that can be written to.\nExiting.\n"
        try:
            os.makedirs(workdir, exist_ok=True)
        except Exception as e:
            log.error(f"Unable to create workdir at {workdir}: {e}.\n{error_message_tail}")
            log.error(e)
            sys.exit(1)
        if not os.access(workdir, os.W_OK):
            log.error(f"Unable to create files at {workdir}.\n{error_message_tail}")
            sys.exit(2)
