#!/usr/bin/env python3
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

"""WSGI Web Server."""
import sys

from lpot.ux.utils.environment import Environment
from lpot.ux.utils.exceptions import NotFoundException
from lpot.ux.utils.logger import change_log_level
from lpot.ux.web.configuration import Configuration
from lpot.ux.web.server import run_server


def main() -> None:
    """Get parameters and initialize server."""
    try:
        configuration = Configuration()
    except NotFoundException as e:
        print(str(e))
        sys.exit(1)

    change_log_level(configuration.log_level)
    print(
        "LPOT UX Server started.\n"
        "Setup port forwarding from "
        f"your local port {configuration.gui_port} to "
        f"{configuration.server_port} on this machine.\n"
        f"Then open address {configuration.get_url()}",
    )

    prepare_environment()
    run_server(configuration)


def prepare_environment() -> None:
    """Prepare environment for LPOT UX."""
    environment = Environment()
    environment.migrate_workloads_list()
    environment.migrate_workloads()
    environment.clean_workloads_wip_status()


if __name__ == "__main__":
    main()
