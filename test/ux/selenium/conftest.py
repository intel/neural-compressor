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
"""Fixtures for selenium test."""

import pytest


def pytest_addoption(parser):
    """Add input arguments."""
    parser.addoption("--address", type=str, required=True)
    parser.addoption("--port", type=str, required=True)
    parser.addoption("--url-prefix", type=str, default="")
    parser.addoption("--token", type=str, required=True)
    parser.addoption("--models-dir", type=str, required=True)


@pytest.fixture
def params(request):
    """Set parameters."""
    params = {}
    params["address"] = request.config.getoption("--address")
    params["port"] = request.config.getoption("--port")
    params["url_prefix"] = request.config.getoption("--url-prefix")
    params["token"] = request.config.getoption("--token")
    params["models_dir"] = request.config.getoption("--models-dir")

    return params
