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

import gevent.monkey

already_patched = gevent.monkey.is_module_patched("threading")
if not already_patched:
    can_patch_ssl = "ssl" not in sys.modules
    gevent.monkey.patch_all(ssl=can_patch_ssl)

from neural_compressor.ux.utils.environment import Environment  # noqa: E402
from neural_compressor.ux.utils.exceptions import NotFoundException  # noqa: E402
from neural_compressor.ux.utils.logger import change_log_level  # noqa: E402
from neural_compressor.ux.web.configuration import Configuration  # noqa: E402
from neural_compressor.ux.web.server import run_server  # noqa: E402


def main() -> None:
    """Get parameters and initialize server."""
    try:
        configuration = Configuration()
    except NotFoundException as e:
        print(str(e))
        sys.exit(1)

    prepare_environment()

    change_log_level(configuration.log_level)
    print(
        "Intel(r) Neural Compressor Bench Server started.\n"
        "Setup port forwarding from "
        f"your local port {configuration.gui_port} to "
        f"{configuration.server_port} on this machine.\n"
        f"Then open address {configuration.get_url()}",
    )

    run_server(configuration)


def prepare_environment() -> None:
    """Prepare environment for Intel® Neural Compressor Bench."""
    environment = Environment()
    environment.ensure_workdir_exists_and_writeable()
    environment.migrate_workloads_list()
    environment.migrate_workloads()
    environment.clean_workloads_wip_status()


if __name__ == "__main__":
    main()
