# -*- coding: utf-8 -*-
# Copyright (c) 2024 Intel Corporation
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

import torch
from packaging.version import Version

# pylint:disable=import-error
try:
    import deepspeed
    import habana_frameworks.torch.hpex

    _hpex_available = True
except:
    _hpex_available = False


def is_hpex_available():
    return _hpex_available


try:
    import intel_extension_for_pytorch as ipex

    _ipex_available = True
except:
    _ipex_available = False


def is_ipex_available():
    return _ipex_available


def get_ipex_version():
    try:
        ipex_version = ipex.__version__.split("+")[0]
    except ValueError as e:  # pragma: no cover
        assert False, "Got an unknown version of intel_extension_for_pytorch: {}".format(e)
    version = Version(ipex_version)
    return version


def get_torch_version():
    try:
        torch_version = torch.__version__.split("+")[0]
    except ValueError as e:  # pragma: no cover
        assert False, "Got an unknown version of torch: {}".format(e)
    version = Version(torch_version)
    return version
