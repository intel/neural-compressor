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
"""UX server utils module."""

import json
import os
import re
import socket
from functools import wraps
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from lpot.ux.utils.exceptions import (
    AccessDeniedException,
    ClientErrorException,
    NotFoundException,
)
from lpot.ux.utils.logger import log
from lpot.ux.utils.proc import Proc

dataset_locations = {
    "tensorflow": {
        "image_recognition": {
            "name": "Imagenet",
            "path": "examples/test/dataset/imagenet",
        },
    },
}

framework_extensions = {
    "tensorflow": ["pb"],
    "onnxrt": ["onnx"],
}

support_boundary_nodes = ["tensorflow"]


def deprecated(func: Callable) -> Any:
    """Signal deprecated function."""

    @wraps(func)
    def report_deprecated(*args: str, **kwargs: str) -> Any:
        log.warning(f"Call to deprecated function {func.__name__}.")
        return func(*args, **kwargs)

    return report_deprecated


def is_hidden(path: str) -> bool:
    """Check if path is for hidden filesystem entry."""
    return "." == os.path.basename(path)[0]


@deprecated
def get_dataset_path(framework: str, domain: str) -> str:
    """Get dataset path for specified framework and domain."""
    dataset = dataset_locations.get(framework, {}).get(domain, {})
    if dataset is None:
        raise Exception("Could not found dataset location.")
    if dataset.get("path") is None:
        raise Exception("Could not found dataset location.")
    return dataset["path"]


def get_framework_from_path(model_path: str) -> Optional[str]:
    """
    Get framework name from model extension.

    :param model_path: Path to model.
    """
    extension = get_file_extension(model_path)
    for framework, extensions in framework_extensions.items():
        if extension in extensions:
            return framework
    return None


def get_file_extension(path: str) -> str:
    """Get file extension without leading dot."""
    return os.path.splitext(path)[1][1:]


def is_model_file(path: str) -> bool:
    """Check if given path is a model of supported framework."""
    return get_framework_from_path(path) is not None


def is_dataset_file(path: str) -> bool:
    """Check if given path is for a dataset of supported framework."""
    dataset_extensions = ["record", "tf_record"]

    extension = get_file_extension(path)

    return extension in dataset_extensions


def get_predefined_config_path(framework: str, domain: str) -> str:
    """Get predefined config for specified model domain."""
    config_path = os.path.join(
        os.path.dirname(__file__),
        "configs",
        "predefined_configs",
        f"{framework}",
        f"{domain}.yaml",
    )
    if config_path and os.path.isfile(config_path):
        return config_path
    raise Exception(
        f"Could not found predefined config for {framework} {domain} model.",
    )


def get_model_zoo_config_path(
    workspace_path: Optional[str],
    framework: str,
    domain: str,
    model_name: str,
    model_dict: Dict[str, Any],
) -> str:
    """Get predefined config for model from Examples."""
    if workspace_path is None:
        return ""
    model_dir = os.path.join(
        workspace_path,
        "examples",
        framework,
        domain,
        model_name,
    )
    yaml_relative_location = model_dict.get("yaml")
    if (
        model_dir
        and yaml_relative_location
        and os.path.exists(os.path.join(model_dir, yaml_relative_location))
    ):
        return os.path.join(model_dir, yaml_relative_location)
    return ""


def get_model_zoo_model_path(
    workspace_path: Optional[str],
    framework: str,
    domain: str,
    model_name: str,
    model_dict: Dict[str, Any],
) -> str:
    """Get path for model from Examples."""
    if workspace_path is None:
        return ""
    model_dir = os.path.join(
        workspace_path,
        "examples",
        framework,
        domain,
        model_name,
    )

    model_relative_path = model_dict.get("download", {}).get("filename", None)
    if (
        model_dir
        and model_relative_path
        and os.path.exists(os.path.join(model_dir, model_relative_path))
    ):
        return os.path.join(model_dir, model_relative_path)
    return ""


def load_json(path: str) -> dict:
    """Load json file and convert it into dict.

    :param path: path to json file
    :type path: str
    :return: dict
    :rtype: dict
    """
    with open(path) as json_data:
        data = json.load(json_data)

    return data


def find_boundary_nodes(model_path: str) -> Dict[str, Any]:
    """Update model's input and output nodes in config file."""
    boundary_nodes: Dict[str, Optional[List[Any]]] = {
        "inputs": None,
        "outputs": None,
    }
    framework = get_framework_from_path(model_path)
    if framework is None:
        raise Exception("Could not find framework for specified model.")
    check_module(framework)
    # Inputs are only required for TF models
    if framework not in support_boundary_nodes:
        return boundary_nodes

    if framework == "tensorflow":
        from lpot.utils.logger import Logger

        Logger().get_logger().setLevel(log.level)
        from lpot.model.model import TensorflowModel

        model = TensorflowModel(model_path)

        inputs = getattr(model, "input_node_names", [])
        outputs = getattr(model, "output_node_names", [])
        outputs += ["custom"]
        boundary_nodes["inputs"] = inputs
        boundary_nodes["outputs"] = list(set(outputs))
        return boundary_nodes
    return {}


def check_module(module_name: str) -> None:
    """Check if module exists. Raise exception when not found."""
    if module_name == "onnxrt":
        module_name = "onnx"
    module = find_spec(module_name)
    if module is None:
        raise ClientErrorException(f"Could not find {module_name} module.")


def get_module_version(module_name: str) -> str:
    """Check module version. Raise exception when not found."""
    version = None
    if module_name == "onnxrt":
        module_name = "onnx"
    command = [
        "python",
        "-c",
        f"import {module_name} as module; print(module.__version__)",
    ]
    proc = Proc()
    proc.run(args=command)
    if proc.is_ok:
        for line in proc.output:
            version = line
    proc.remove_logs()
    if version is None:
        raise ClientErrorException(f"Could not found version of {module_name} module.")
    return version


def get_size(path: str, unit: str = "MB", add_unit: bool = False) -> Union[str, int]:
    """Check file or directory size."""
    supported_units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
    }
    unit_modifier = supported_units.get(unit, None)
    if unit_modifier is None:
        raise Exception(
            "Unit not supported. Select one of following: "
            + str(supported_units.keys()),
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


def load_model_config() -> Dict[str, Any]:
    """Load model configs from json."""
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "configs",
            "models.json",
        ),
        "r",
    ) as f:
        models_config = json.load(f)
    if isinstance(models_config, dict):
        return models_config
    return {}


def load_dataloader_config() -> List[Dict[str, Any]]:
    """Load dataloader configs from json."""
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "configs",
            "dataloaders.json",
        ),
        "r",
    ) as f:
        dataloaders_config = json.load(f)
    if isinstance(dataloaders_config, list):
        return dataloaders_config
    return []


def load_transforms_config() -> List[Dict[str, Any]]:
    """Load dataloader configs from json."""
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "configs",
            "transforms.json",
        ),
        "r",
    ) as f:
        dataloaders_config = json.load(f)
    if isinstance(dataloaders_config, list):
        return dataloaders_config
    return []


def load_help_lpot_params(parameter: str) -> Dict[str, Any]:
    """Load help info from json for metrics, objectives and strategies."""
    with open(
        os.path.join(os.path.dirname(__file__), "configs", f"{parameter}.json"),
        "r",
    ) as f:
        config = json.load(f)
    if isinstance(config, dict):
        return config
    else:
        return {}


def replace_with_values(param: dict, file_path: str) -> None:
    """Replace parameters with value."""
    with open(file_path, "r+") as opened_file:
        text = opened_file.read()
        for key, value in param.items():
            key_to_search = "".join(["{{", key, "}}"])
            text = re.sub(key_to_search, value, text)
        opened_file.seek(0)
        opened_file.write(text)


def verify_file_path(path: str) -> None:
    """Check if path can be accessed."""
    restricted_paths = [
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/lib",
        "/media",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/snap",
        "/srv",
        "/swapfile",
        "/usr",
        "/var",
    ]
    real_path = os.path.realpath(path)
    if not os.path.exists(real_path):
        raise NotFoundException("File not found.")
    if os.stat(real_path).st_uid == 0:
        raise AccessDeniedException("Access denied.")
    for path_element in real_path.split(os.sep):
        if path_element.startswith("."):
            raise AccessDeniedException("Access denied.")
    for restricted_path in restricted_paths:
        if real_path.startswith(restricted_path):
            raise AccessDeniedException("Access denied.")


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


def is_development_env() -> bool:
    """Return true if LPOT_MODE is development else false."""
    return os.environ.get("LPOT_MODE") == "development"
