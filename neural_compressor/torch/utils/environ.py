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
"""Intel Neural Compressor PyTorch environment check."""

import importlib
import sys

import torch
from packaging.version import Version


################ Check imported sys.module first to decide behavior #################
def is_ipex_imported() -> bool:
    """Check whether intel_extension_for_pytorch is imported."""
    for name, _ in sys.modules.items():
        if name == "intel_extension_for_pytorch":
            return True
    return False


def is_transformers_imported() -> bool:
    """Check whether transformers is imported."""
    for name, _ in sys.modules.items():
        if name == "transformers":
            return True
    return False


################ Check available sys.module to decide behavior #################
def is_package_available(package_name):
    """Check if the package exists in the environment without importing.

    Args:
        package_name (str): package name
    """
    from importlib.util import find_spec

    package_spec = find_spec(package_name)
    return package_spec is not None


## check hpex
if is_package_available("habana_frameworks"):
    _hpex_available = True
    import habana_frameworks.torch.hpex  # pylint: disable=E0401
else:
    _hpex_available = False


def is_hpex_available():
    """Returns whether hpex is available."""
    return _hpex_available


## check ipex
if is_package_available("intel_extension_for_pytorch"):
    _ipex_available = True
else:
    _ipex_available = False


def is_ipex_available():
    """Return whether ipex is available."""
    return _ipex_available


def get_ipex_version():
    """Return ipex version if ipex exists."""
    if is_ipex_available():
        try:
            import intel_extension_for_pytorch as ipex

            ipex_version = ipex.__version__.split("+")[0]
        except ValueError as e:  # pragma: no cover
            assert False, "Got an unknown version of intel_extension_for_pytorch: {}".format(e)
        version = Version(ipex_version)
        return version
    else:
        return None


TORCH_VERSION_2_2_2 = Version("2.2.2")


def get_torch_version():
    """Return torch version if ipex exists."""
    try:
        torch_version = torch.__version__.split("+")[0]
    except ValueError as e:  # pragma: no cover
        assert False, "Got an unknown version of torch: {}".format(e)
    version = Version(torch_version)
    return version


GT_TORCH_VERSION_2_3_2 = get_torch_version() > Version("2.3.2")


def get_accelerator(device_name="auto"):
    """Return the recommended accelerator based on device priority."""
    global accelerator  # update the global accelerator when calling this func
    from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator

    accelerator = auto_detect_accelerator(device_name)
    return accelerator


# for direct user access, used by @device_synchronize, can be changed by set_accelerator
accelerator = get_accelerator()


# for habana ease-of-use
def device_synchronize(raw_func):
    """Function decorator that calls accelerated.synchronize before and after a function call."""
    from functools import wraps

    @wraps(raw_func)
    def new_func(*args, **kwargs):
        accelerator.synchronize()
        output = raw_func(*args, **kwargs)
        accelerator.synchronize()
        return output

    return new_func
