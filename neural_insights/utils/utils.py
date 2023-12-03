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
"""Utils module for Neural Insights server."""
import os
import socket
from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Union

from neural_insights.utils.exceptions import ClientErrorException, NotFoundException


def determine_ip() -> str:
    """Return IP to be used by server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.0.0.0", 1))
        ip = sock.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        sock.close()

    return ip


def get_size(path: str, unit: str = "MB", add_unit: bool = False) -> Union[str, int]:
    """Check file or directory size."""
    supported_units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }
    unit_modifier = supported_units.get(unit, None)
    if unit_modifier is None:
        raise Exception(
            "Unit not supported. Select one of following: " + str(supported_units.keys()),
        )
    root_dir = Path(path)
    if root_dir.is_file():
        size = root_dir.stat().st_size
    else:
        size = sum(f.stat().st_size for f in root_dir.glob("**/*") if f.is_file())
    size = int(round(size / unit_modifier))
    if add_unit:
        return f"{size}{unit}"
    return size


def check_module(module_name: str) -> None:
    """Check if module exists.

    Raise exception when not found.
    """
    if module_name == "onnxrt":
        module_name = "onnxruntime"
    if module_name == "pytorch":
        module_name = "torch"
    module = find_spec(module_name.lower())
    if module is None:
        raise ClientErrorException(f"Could not find {module_name} module.")


def get_file_extension(path: str) -> str:
    """Get file extension without leading dot."""
    return os.path.splitext(path)[1][1:]


def get_framework_from_path(model_path: str) -> Optional[str]:
    """Get framework name from model extension.

    :param model_path: Path to model.
    """
    from neural_insights.components.model.repository import ModelRepository

    model_repository = ModelRepository()
    try:
        model = model_repository.get_model(model_path)
        return model.get_framework_name()
    except NotFoundException:
        return None
