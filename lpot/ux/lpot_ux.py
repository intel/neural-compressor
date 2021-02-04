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

import argparse
import socket

from lpot.ux.web.server import run_server


def get_server_ip() -> str:
    """Return IP used by server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.0.0.0", 1))
        ip = sock.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        sock.close()

    return ip


def main() -> None:
    """Get parameters and initialize server."""
    address = get_server_ip()

    parser = argparse.ArgumentParser(description="Run LPOT-UX server.")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=5000,
        help="port number to listen on",
    )
    args = parser.parse_args()
    port = args.port

    print(f"Server listening on http://{address}:{port}")

    run_server(address, port)


if __name__ == "__main__":
    main()
