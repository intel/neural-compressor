#!/usr/bin/env python3
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
"""WSGI Web Server."""
import sys

import gevent.monkey

already_patched = gevent.monkey.is_module_patched("threading")
if not already_patched:
    can_patch_ssl = "ssl" not in sys.modules
    gevent.monkey.patch_all(ssl=can_patch_ssl)

from neural_insights.utils.exceptions import NotFoundException  # noqa: E402
from neural_insights.utils.logger import change_log_level, log  # noqa: E402
from neural_insights.web.configuration import Configuration  # noqa: E402
from neural_insights.web.server import run_server  # noqa: E402


def main() -> None:
    """Get parameters and initialize server."""
    try:
        configuration = Configuration()
    except NotFoundException as e:
        print(str(e))
        sys.exit(1)

    change_log_level(configuration.log_level)

    log.info("Neural Insights Server started.\n")

    if configuration.allow_insecure_connections:
        log.warning(
            "Running in insecure mode.\n" "Everyone in your network may attempt to access this server.\n",
        )

    log.info(f"Open address {configuration.get_url()}")
    configuration.dump_token_to_file()

    run_server(configuration)


if __name__ == "__main__":
    main()
